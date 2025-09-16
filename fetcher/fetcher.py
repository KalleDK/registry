#!/usr/bin/env python3

import contextlib
from datetime import datetime
import pathlib
import re
import shutil
import subprocess
import typing
import httpx
import pydantic
import typer
import tempfile
import tomllib

class Settings(pydantic.BaseModel):
    name: str
    release: int = 1
    arch: str
    description: str
    url: str
    maintainer: str
    license: str
    bindir: str = "/usr/bin"
    binaries: list[str]
    gh_repo: str
    gh_asset_regex: str
    strip_components: int = 0
    pkgtype: str = "deb"

class GithubAsset(pydantic.BaseModel):
    url: str
    id: int
    name: str
    label: str | None
    content_type: str
    size: int
    download_count: int
    created_at: datetime
    updated_at: datetime
    browser_download_url: str

class GitHubRelease(pydantic.BaseModel):
    url: str
    id: int
    name: str
    draft: bool
    prerelease: bool
    created_at: datetime
    published_at: datetime
    assets: list[GithubAsset]


def get_releases(gh_repo: str):
    url = f"https://api.github.com/repos/{gh_repo}/releases"
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    response = httpx.get(url, headers=headers, follow_redirects=True)
    response.raise_for_status()
    return [GitHubRelease.model_validate(release) for release in response.json()] # type: ignore

def deb_filename(name: str, version: str, arch: str):
    return f"{name}_{version}_{arch}.deb"

def apk_filename(name: str, version: str, arch: str):
    return f"{name}-{version}-{arch}.apk"

def deb_package_releases(name: str, license: str, version: str, arch: str, description: str, url: str, maintainer: str, binaries: list[str], bindir: str, cwd: pathlib.Path):
    filename = deb_filename(name, version, arch)
    subprocess.run([
        "fpm",
        "-s", "dir",
        "-t", "deb", 
        "--name", name,
        "--version", version,
        "--architecture", arch,
        "--license", license,
        "--url", url,
        "--maintainer", maintainer,
        "--description", description,
        "-p", filename,
        *[f"{binary}={bindir}/{binary}" for binary in binaries]
        ], cwd=cwd)
    return cwd / filename

def apk_package_releases(name: str, license: str, version: str, arch: str, description: str, url: str, maintainer: str, binaries: list[str], bindir: str, cwd: pathlib.Path):
    filename = apk_filename(name, version, arch)
    subprocess.run([
        "fpm",
        "-s", "dir",
        "-t", "apk", 
        "--name", name,
        "--version", version,
        "--architecture", arch,
        "--license", license,
        "--url", url,
        "--maintainer", maintainer,
        "--description", description,
        "-p", filename,
        *[f"{binary}={bindir}/{binary}" for binary in binaries]
        ], cwd=cwd)
    return cwd / filename
    
def download_asset(asset: GithubAsset, filename: pathlib.Path):
    response = httpx.get(asset.browser_download_url, follow_redirects=True)
    response.raise_for_status()
    filename.write_bytes(response.content)
    return filename

def unpack_file(filename: pathlib.Path, strip_components: int):
    p = subprocess.run(["tar", "-xzv", f"--strip-components={strip_components}", "-f", f"{filename}"], cwd=filename.parent)
    p.check_returncode()

def load_settings(config: pathlib.Path):
    return Settings.model_validate(tomllib.loads(config.read_text())) # type: ignore


def debian_arch(arch: str):
    return {
        "x86_64": "amd64",
        "i386": "i386",
        "aarch64": "arm64",
        "armv7l": "armhf",
        "armv6l": "armhf"
    }.get(arch, arch)

def apk_arch(arch: str):
    return {
        "x86_64": "x86_64",
        "i386": "x86",
        "i686": "x86",
        "arm64": "aarch64"
    }.get(arch, arch)

@contextlib.contextmanager
def tempdir(path: pathlib.Path | None, delete: bool):
    if path is None:
        with tempfile.TemporaryDirectory(delete=delete) as tmpdirname:
            base = pathlib.Path(tmpdirname)
            print(f"Created temporary directory: {base}")
            yield base
    else:
        path_exists = path.exists()
        try:
            if not path_exists:
                path.mkdir()
            print(f"Created temporary directory: {path}")
            yield path
        finally:
            if not path_exists and delete:
                shutil.rmtree(path)
                

def main(config: pathlib.Path = pathlib.Path("fetcher.toml"), tmpdir: typing.Optional[pathlib.Path] = None, delete: bool = True):
    settings = load_settings(config)
    print(settings)
    with tempdir(tmpdir, delete) as base:
        print(f"Created temporary directory: {base}")
        releases = get_releases(settings.gh_repo)
        releases = [release for release in releases if not release.draft and not release.prerelease]
        release = releases[0]
        asset_re = re.compile(settings.gh_asset_regex)
        assets = [asset for asset in release.assets if asset_re.match(asset.name)]
        if settings.pkgtype == "deb":
            deb_name = deb_filename(settings.name, release.name.replace("v",""), debian_arch(settings.arch))
            if (pathlib.Path.cwd() / deb_name).exists():
                print(f"Package {deb_name} already exists. Skipping.")
                return
            for asset in assets:
                print(f"Found asset: {asset.name}")
                filename = base / asset.name
                download_asset(asset, filename)
                unpack_file(filename, settings.strip_components)
            deb = deb_package_releases(settings.name, settings.license, release.name.replace("v",""), debian_arch(settings.arch), settings.description, settings.url, settings.maintainer, settings.binaries, "/usr/bin", base)
            print(f"Created package: {deb}")
            shutil.move(deb, pathlib.Path.cwd())
        if settings.pkgtype == "apk":
            apk_name = apk_filename(settings.name, release.name.replace("v",""), apk_arch(settings.arch))
            if (pathlib.Path.cwd() / apk_name).exists():
                print(f"Package {apk_name} already exists. Skipping.")
                return
            for asset in assets:
                print(f"Found asset: {asset.name}")
                filename = base / asset.name
                download_asset(asset, filename)
                unpack_file(filename, settings.strip_components)
            apk = apk_package_releases(settings.name, settings.license, release.name.replace("v",""), apk_arch(settings.arch), settings.description, settings.url, settings.maintainer, settings.binaries, "/usr/bin", base)
            print(f"Created package: {apk}")
            shutil.move(apk, pathlib.Path.cwd())


if __name__ == "__main__":
    typer.run(main)
