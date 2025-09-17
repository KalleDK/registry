#!/usr/bin/env python3

import contextlib
import dataclasses
import http
import logging
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import typing
from datetime import datetime
from typing import Annotated
from zoneinfo import ZoneInfo

import httpx
import pydantic
import tomllib
import typer
import yaml
from rich.console import Console

_log = logging.getLogger(__name__)

# region Settings


class Settings(pydantic.BaseModel):
    name: str
    arch: str
    description: str
    url: str
    maintainer: str
    license: str
    bindir: str = "/usr/bin"
    binaries: list[str]
    section: str | None = None
    gh_repo: str
    gh_asset_regex: re.Pattern[str]
    strip_components: int = 0
    pkgtypes: list[str]
    depends: list[str] = pydantic.Field(default_factory=list[str])

    @pydantic.field_validator("depends", mode="before")
    @classmethod
    def validate_depends(cls, v: str | list[str] | None) -> list[str]:
        if isinstance(v, str):
            return [x.strip() for x in v.split(",")]
        if v is None:
            return []
        return v

    @pydantic.field_validator("pkgtypes", mode="before")
    @classmethod
    def validate_packages(cls, v: str | list[str] | None) -> list[str] | None:
        if isinstance(v, str):
            return [x.strip() for x in v.split(",")]
        return v

    @pydantic.field_validator("gh_asset_regex", mode="before")
    @classmethod
    def validate_gh_asset_regex(cls, v: str | re.Pattern[str]) -> re.Pattern[str]:
        if isinstance(v, str):
            return re.compile(v)
        return v

    @classmethod
    def from_file(cls, path: str | pathlib.Path) -> "Settings":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls.model_validate(data)


# endregion


# region GitHub API models

URL = typing.NewType("URL", str)
UserID = typing.NewType("UserID", int)
ReleaseID = typing.NewType("ReleaseID", int)
AssetID = typing.NewType("AssetID", int)


class GithubUser(pydantic.BaseModel):
    login: str
    id: UserID
    node_id: str
    avatar_url: URL
    gravatar_id: str
    url: URL
    html_url: URL
    followers_url: URL
    following_url: URL
    gists_url: URL
    starred_url: URL
    subscriptions_url: URL
    organizations_url: URL
    repos_url: URL
    events_url: URL
    received_events_url: URL
    type: str
    user_view_type: str
    site_admin: bool


class GithubAsset(pydantic.BaseModel):
    url: URL
    id: AssetID
    node_id: str
    name: str
    label: str
    uploader: GithubUser
    content_type: str
    state: str
    size: int
    download_count: int
    created_at: datetime
    updated_at: datetime
    browser_download_url: URL


class GithubReactions(pydantic.BaseModel):
    url: str
    total_count: int
    positive: int = pydantic.Field(..., alias="+1")
    negative: int = pydantic.Field(..., alias="-1")
    laugh: int
    hooray: int
    confused: int
    heart: int
    rocket: int
    eyes: int


class GithubRelease(pydantic.BaseModel):
    url: str
    assets_url: str
    upload_url: str
    html_url: str
    id: int
    author: GithubUser
    node_id: str
    tag_name: str
    target_commitish: str
    name: str
    draft: bool
    immutable: bool
    prerelease: bool
    created_at: datetime
    updated_at: datetime
    published_at: datetime
    assets: list[GithubAsset]
    tarball_url: str
    zipball_url: str
    body: str
    reactions: GithubReactions

    def get_asset_by_regex(self, pattern: re.Pattern[str]) -> GithubAsset:
        assets = [a for a in self.assets if pattern.match(a.name)]
        if len(assets) == 0:
            raise ValueError(f"No asset found matching pattern: {pattern.pattern}")
        if len(assets) > 1:
            raise ValueError(
                f"Multiple assets found matching pattern: {pattern.pattern}"
            )
        return assets[0]


class GithubHeader(pydantic.BaseModel):
    last_modified: str = pydantic.Field(alias="last-modified")
    etag: str | None = pydantic.Field(alias="etag", default=None)

    @pydantic.computed_field
    @property
    def modified_at(self) -> datetime:
        return datetime.strptime(
            self.last_modified, r"%a, %d %b %Y %H:%M:%S %Z"
        ).replace(tzinfo=ZoneInfo("GMT"))


class GithubCacheData[DataT](pydantic.BaseModel):
    etag: str | None
    last_modified: str
    modified_at: datetime
    data: DataT


class GithubCacheFile[TData](pydantic.BaseModel):
    path: pathlib.Path

    def load(self, clss: type[TData]) -> GithubCacheData[TData]:
        return (
            pydantic.RootModel[GithubCacheData[clss]]
            .model_validate_json(self.path.read_text())
            .root
        )

    def save(self, data: GithubCacheData[TData]):
        self.path.write_text(data.model_dump_json(indent=2, by_alias=True))

    def exists(self) -> bool:
        return self.path.exists()


@dataclasses.dataclass
class GHClient:
    _client: httpx.Client
    cache: pathlib.Path

    def _head(self, url: str) -> GithubHeader:
        response = self._client.head(url)
        response.raise_for_status()
        return GithubHeader.model_validate(response.headers)

    def _get(self, url: str) -> httpx.Response:
        response = self._client.get(url)
        response.raise_for_status()
        return response

    def _get_cached[TData](
        self,
        clss: type[TData],
        url: str,
        filename: str | None = None,
        force: bool = False,
    ) -> tuple[TData, bool]:
        filename = filename or re.sub(r"[^a-zA-Z0-9]", "_", url)
        cache_file = GithubCacheFile[clss](path=self.cache / filename)

        req = self._client.build_request("GET", url)

        if cache_file.exists() and not force:
            _log.info(f"Cache file exists: {cache_file}")
            cached = cache_file.load(clss)
            _log.debug(f"Cached Last-Modified: {cached.last_modified}")
            req.headers["If-Modified-Since"] = cached.last_modified
        else:
            _log.info(f"Cache file does not exist: {cache_file}")
            cached = None

        resp = self._client.send(req)
        if resp.status_code == http.HTTPStatus.NOT_MODIFIED and cached is not None:
            _log.info(f"Cache hit for {url}")
            return cached.data, True

        resp = self._client.send(req)
        resp.raise_for_status()
        headers = GithubHeader.model_validate(resp.headers)
        _log.debug(f"Response Last-Modified: {headers.last_modified}")
        data = pydantic.RootModel[clss].model_validate_json(resp.read()).root

        _log.info(f"Cache miss for {url}, saving to {cache_file} [force={force}]")
        cache_file.save(
            GithubCacheData[clss](
                etag=headers.etag,
                modified_at=headers.modified_at,
                last_modified=headers.last_modified,
                data=data,
            )
        )

        return data, False

    @classmethod
    def create(cls, token: str, cache_dir: pathlib.Path) -> "GHClient":
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
        headers = {"Authorization": f"Bearer {token}"}
        client = httpx.Client(headers=headers, follow_redirects=True)
        return cls(client, cache_dir)

    def get_latest_release(
        self, gh_repo: str, force: bool = False
    ) -> tuple[GithubRelease, bool]:
        url = f"https://api.github.com/repos/{gh_repo}/releases/latest"
        return self._get_cached(
            GithubRelease,
            url,
            filename=f"{gh_repo.replace('/', '_')}_latest.json",
            force=force,
        )

    def download_asset(self, asset: GithubAsset, dir: pathlib.Path) -> pathlib.Path:
        dst = dir / asset.name
        response = self._client.get(asset.browser_download_url)
        response.raise_for_status()
        dst.write_bytes(response.content)
        return dst


# endregion


# region nFPM


class NFPMContents(pydantic.BaseModel):
    src: str
    dst: str


class NFPMConfig(pydantic.BaseModel):
    name: str
    arch: str
    platform: str
    version: str
    maintainer: str
    description: str
    homepage: str
    license: str
    section: str | None
    depends: list[str] | None
    contents: list[NFPMContents]


# endregion

app = typer.Typer(help="Fetch and package binaries from GitHub releases")

console = Console()


def unpack(path: pathlib.Path, dest: pathlib.Path, strip_components: int = 0):
    p = subprocess.run(
        [
            "tar",
            "-xzv",
            f"--strip-components={strip_components}",
            "-f",
            f"{path}",
            "-C",
            f"{dest}",
        ],
        capture_output=True,
    )
    p.check_returncode()
    _log.info(f"Unpacked {path} to {dest}")
    for line in p.stdout.decode().splitlines():
        _log.debug(f"Unpack: {line}")


@contextlib.contextmanager
def tempdirs(dir: pathlib.Path | None = None, delete: bool = True):
    if dir is not None:
        dir.mkdir(parents=True, exist_ok=True)
        yield dir
    else:
        with tempfile.TemporaryDirectory(dir=dir, delete=delete) as tmpdir:
            yield pathlib.Path(tmpdir)


def get_arch(arch: str, target: str):
    if arch == "x86_64":
        if target == "apk":
            return "x86_64"
        return "amd64"
    if arch == "aarch64":
        if target == "apk":
            return "aarch64"
        return "arm64"
    return arch


@dataclasses.dataclass
class FetcherSession:
    gh: GHClient
    base: pathlib.Path
    settings: Settings
    tmpbase: pathlib.Path | None = None

    def build_packages(self, force: bool = False):
        release, is_cached = self.gh.get_latest_release(
            self.settings.gh_repo, force=force
        )
        if is_cached:
            return None

        asset = release.get_asset_by_regex(self.settings.gh_asset_regex)
        _log.info(f"Found asset: {asset.name} ({asset.size} bytes)")
        return self._build_packages(asset, release)

    def _build_packages(self, asset: GithubAsset, release: GithubRelease):
        with tempdirs(dir=self.tmpbase) as tmpdir:
            # region Create skeleton
            src_dir = tmpdir / "src"
            if not src_dir.exists():
                src_dir.mkdir()

            asset_dir = tmpdir / "assets"
            if not asset_dir.exists():
                asset_dir.mkdir()

            dist_dir = tmpdir / "dist"
            if not dist_dir.exists():
                dist_dir.mkdir()

            # endregion

            # Download Asset
            asset_path = self.gh.download_asset(asset, src_dir)
            # asset_path = src_dir / "uv-x86_64-unknown-linux-gnu.tar.gz"

            # Unpack Asset
            unpack(asset_path, asset_dir, self.settings.strip_components)

            # Create nFPM config

            packages: list[pathlib.Path] = []

            for pkgtype in self.settings.pkgtypes:
                nfpm_config_path = tmpdir / f"nfpm_{pkgtype}.yaml"
                nfpm_conf = NFPMConfig(
                    name=self.settings.name,
                    arch=get_arch(self.settings.arch, pkgtype),
                    platform="linux",
                    version=release.tag_name.replace("v", ""),
                    maintainer=self.settings.maintainer,
                    description=self.settings.description,
                    homepage=self.settings.url,
                    license=self.settings.license,
                    section=self.settings.section,
                    depends=self.settings.depends,
                    contents=[
                        NFPMContents(src=f"./assets/{binary}", dst=f"/usr/bin/{binary}")
                        for binary in self.settings.binaries
                    ],
                )
                nfpm_config_path.write_text(
                    yaml.dump(
                        nfpm_conf.model_dump(
                            exclude_none=True, by_alias=True, mode="json"
                        )
                    )
                )

                p = subprocess.run(
                    [
                        "nfpm",
                        "pkg",
                        "-f",
                        nfpm_config_path.name,
                        "-p",
                        pkgtype,
                        "-t",
                        f"{dist_dir.name}",
                    ],
                    cwd=tmpdir,
                    capture_output=True,
                )
                p.check_returncode()

                m = re.search(r"created package: dist/([^ \n]+)", p.stdout.decode())
                if m:
                    packages.append(dist_dir / m.group(1))
                else:
                    raise ValueError("Could not find created package in nfpm output")

            for pkg in packages:
                _log.debug(f"Created package: {pkg}")
                shutil.move(pkg, self.base / pkg.name)

            return [self.base / pkg.name for pkg in packages]


@dataclasses.dataclass
class Fetcher:
    base: pathlib.Path
    gh: GHClient
    tmpbase: pathlib.Path | None = None

    @classmethod
    def create(
        cls,
        base: pathlib.Path,
        tmpbase: pathlib.Path | None = None,
        token: str | None = None,
    ):
        if token is None:
            token = os.environ.get("GITHUB_TOKEN")
        if token is None:
            raise ValueError("GITHUB_TOKEN environment variable not set")
        gh = GHClient.create(token, base / ".cache")
        return cls(base, gh, tmpbase)

    def build_packages(self, config: pathlib.Path, force: bool = False):
        settings = Settings.from_file(config)
        session = FetcherSession(self.gh, config.parent, settings, self.tmpbase)
        return session.build_packages(force=force)


@app.command("fetch")
def fetch_binaries(
    name: Annotated[str | None, typer.Argument()] = None,
    all_pkgs: Annotated[bool, typer.Option("--all", "-a")] = False,
    path: Annotated[pathlib.Path, typer.Option()] = pathlib.Path("pkgs"),
    config: Annotated[str | None, typer.Option("--config", "-c")] = None,
    force: Annotated[bool, typer.Option("--force", "-f")] = False,
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True)] = 0,
    temp: bool = False,
):
    # region Logging
    if verbose >= 3:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("httpcore").setLevel(logging.INFO)
    if verbose == 2:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
    elif verbose == 1:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
    else:
        logging.basicConfig(level=logging.WARNING)
    # endregion

    if config is None:
        config_name = "fetcher.toml"
    else:
        config_name = f"fetcher.{config}.toml"

    if all_pkgs and name is not None:
        raise ValueError("Cannot use --all and specify a package name")

    tmp = path / ".tmp" if temp else None

    fetcher = Fetcher.create(path, tmp)

    if name is not None:
        pkgs = fetcher.build_packages(path / name / config_name, force)
        console.print(f"Created packages: {pkgs}")

    if all_pkgs:
        print(f"Processing all packages in {path}")
        files = list(path.glob("**/fetcher*.toml"))
        print(f"Found {len(files)} package configurations")
        for file in files:
            console.print(f"Processing {file}")
            Settings.from_file(file)  # validate settings

        for file in files:
            pkgs = fetcher.build_packages(file, force)
            console.print(f"Created packages: {pkgs}")


if __name__ == "__main__":
    app()
