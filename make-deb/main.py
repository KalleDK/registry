
import base64
import dataclasses
import enum
import gzip
import hashlib
import logging
import os
import pathlib
import lzma
import re
import shutil
import subprocess
from typing import NamedTuple, NewType


logger = logging.getLogger(__name__)


class Arch(enum.Enum):
    amd64 = "amd64"
    arm64 = "arm64"
    i386 = "i386"

Dist = NewType("Dist", str)
Component = NewType("Component", str)

class DistComponent(NamedTuple):
    dists: list[Dist]
    component: Component

class DistComponentArch(NamedTuple):
    dist: Dist
    component: Component
    arch: Arch

ARCH_RE = re.compile(r"Architecture: (.*)")
PACKAGE_RE = re.compile(r"Package: (.*)")
FILENAME_RE = re.compile(r"Filename: (.*)") 

@dataclasses.dataclass
class PoolDeb:
    name: str
    arch: Arch
    component: Component
    path: pathlib.Path
    info: str

@dataclasses.dataclass
class DebPackageSrc:
    name: str
    arch: Arch
    dist_component: DistComponent
    path: pathlib.Path

    @classmethod
    def from_file(cls, path: pathlib.Path, dist_component: DistComponent):
        p = subprocess.run(["dpkg-scanpackages", str(path)], capture_output=True)
        p.check_returncode()
        output = p.stdout.decode("utf-8")
        m = ARCH_RE.search(output)
        if not m:
            raise ValueError("No architecture found")
        arch = Arch(m.group(1))
        m = PACKAGE_RE.search(output)
        if not m:
            raise ValueError("No package name found")
        return cls(m.group(1), arch, dist_component, path)
        

def find_packages(path: pathlib.Path, dist_component: DistComponent) -> list[DebPackageSrc]:
    return [DebPackageSrc.from_file(p, dist_component) for p in path.glob("**/*.deb")]

@dataclasses.dataclass
class HashedFile:
    path: pathlib.Path
    data: bytes
    size: int = dataclasses.field(init=False)
    md5: str = dataclasses.field(init=False)
    sha1: str = dataclasses.field(init=False)
    sha256: str = dataclasses.field(init=False)

    def __post_init__(self):
        self.size = len(self.data)
        self.md5 = hashlib.md5(self.data).hexdigest()
        self.sha1 = hashlib.sha1(self.data).hexdigest()
        self.sha256 = hashlib.sha256(self.data).hexdigest()
    
    def write(self):
        self.path.write_bytes(self.data)
        return self

@dataclasses.dataclass
class PackageFile(HashedFile):
    pass

@dataclasses.dataclass
class GZPackageFile(HashedFile):
    pass

@dataclasses.dataclass
class XZPackageFile(HashedFile):
    pass

@dataclasses.dataclass
class ReleaseFile(HashedFile):
    pass

def create_static_indexes(path: pathlib.Path, header: str = "", base: pathlib.Path | None = None):
    base = base or path
    files: list[pathlib.Path] = list(p for p in path.iterdir() if p.name != "index.md")
    for p in files:
        if p.is_dir():
            create_static_indexes(p, header=header, base=base)
    
    filelinks = sorted(list(f" - [🗁 {p.name}]({p.name})" for p in path.iterdir() if p.is_dir()))
    filelinks.extend(sorted(list(f" - [🗋 {p.name}]({p.name})" for p in path.iterdir() if p.is_file())))
    files_str = "\n".join(filelinks)
    rel_path = path.relative_to(base.parent)
    parts = rel_path.parts
    link_parts = parts[:-1]
    
    link_parts = [f"[{p}](" + ("../" * (len(parts) - n)) + ") / " for n, p in enumerate(link_parts, 1)]
    print(link_parts)


    navline = "/ " + "".join(link_parts) + parts[-1]
    
    INDEX = f"""
{header}

{navline}

## Files:
{files_str}
"""
    (path / "index.md").write_text(INDEX)


@dataclasses.dataclass
class Repo:
    path: pathlib.Path
    dcas: set[DistComponentArch] = dataclasses.field(default_factory=set)
    packages: dict[Dist, dict[Component, dict[Arch, list[PoolDeb]]]] = dataclasses.field(default_factory=dict)

    def get_pool_path(self, deb: DebPackageSrc, component: Component) -> pathlib.Path:
        return self.path / "pool" / component / deb.path.name

    
        


    def add_binary(self, deb: DebPackageSrc):
        component = deb.dist_component.component
        pool_path = self.get_pool_path(deb, component)
        if not pool_path.parent.exists():
            logger.info(f"Creating directory {pool_path.parent}")
            pool_path.parent.mkdir(parents=True, exist_ok=True)
        if not pool_path.exists():
            shutil.copy(deb.path, pool_path)
        logger.info(f"Added {deb.path} to {pool_path}")
        p = subprocess.run(["dpkg-scanpackages", str(pool_path)], check=True, capture_output=True)
        info = p.stdout.decode("utf-8")
        info = FILENAME_RE.sub(f"Filename: {pool_path.relative_to(self.path)}", info)
        for dist in  deb.dist_component.dists:
            dca = DistComponentArch(dist, component, deb.arch)
            self.dcas.add(dca)
            pool_deb = PoolDeb(deb.name, deb.arch, component, pool_path, info)
            self.packages.setdefault(dist, {}).setdefault(component, {}).setdefault(deb.arch, []).append(pool_deb)
            
    
    def _create_gzip(self, path: pathlib.Path, pkg: PackageFile):
        file = GZPackageFile(path, gzip.compress(pkg.data))
        file.write()
        return file
    
    def _create_xz(self, path: pathlib.Path, pkg: PackageFile):
        file = XZPackageFile(path, lzma.compress(pkg.data))
        file.write()
        return file
    
    def _create_binary_packages_file(self, path: pathlib.Path, packages: list[PoolDeb]):
        data =  ("".join(package.info for package in packages)).encode("utf-8")
        file = PackageFile(path, data)
        file.write()
        return file
    
    def _create_binary_release_file(self, path: pathlib.Path, dist: Dist, component: Component, arch: Arch):
        data = f"""Archive: {dist}
Origin: Debian
Label: Debian
Version: 12.7
Acquire-By-Hash: yes
Component: {component}
Architecture: {arch.value}""".encode("utf-8")
        file = PackageFile(path, data)
        file.write()
        return file

    def create_dist(self, base_path: pathlib.Path, dist: Dist):
        path = base_path / dist
        path.mkdir(parents=True, exist_ok=True)
        files: list[HashedFile] = []
        for component in self.packages[dist]:
            files.extend(self.create_component(path, dist, component))
        
        components: set[Component] = set(self.packages[dist].keys())

        archs: set[Arch] = set()
        for component in self.packages[dist]:
            archs.update(self.packages[dist][component].keys())


        p = subprocess.run(["date", "-Ru"], capture_output=True, check=True)
        dateline = p.stdout.decode("utf-8").strip()
        md5sums = '\n'.join(f" {p.md5} {p.size} {p.path.relative_to(path)}" for p in files)
        sha1sums = '\n'.join(f" {p.sha1} {p.size} {p.path.relative_to(path)}" for p in files)
        sha256sums = '\n'.join(f" {p.sha256} {p.size} {p.path.relative_to(path)}" for p in files)

        release = f"""
Origin: KalleDK
Label: Debian
Codename: {dist}
Version: 12.3
Architectures: {', '.join(arch.value for arch in archs)}
Components: {', '.join(components)}
Description: Repository for KalleDK releases
Date: {dateline}
MD5Sum:
{md5sums}
SHA1:
{sha1sums}
SHA256:
{sha256sums}"""
        
        
        release_file = ReleaseFile(path / "Release", release.encode("utf-8"))
        release_file.write()
    
        return release_file

    def create_component(self, base_path: pathlib.Path, dist: Dist, component: Component):
        path = base_path / component
        path.mkdir(parents=True, exist_ok=True)

        files: list[HashedFile] = []
        
        for arch in self.packages[dist][component]:
            files.extend(self.create_binary_release(path, dist, component, arch))

        return files

    def create_binary_release(self, base_path: pathlib.Path, dist: Dist, component: Component, arch: Arch):
        path = base_path / f"binary-{arch.value}"
        path.mkdir(parents=True, exist_ok=True)
        
        release_file = self._create_binary_release_file(path / "Release", dist, component, arch)
        pkg_file = self._create_binary_packages_file(path / "Packages", self.packages[dist][component][arch])
        pkg_gz_file = self._create_gzip(path / "Packages.gz", pkg_file)
        pkg_xz_file = self._create_xz(path / "Packages.xz", pkg_file)

        files: list[HashedFile] = [release_file, pkg_file, pkg_gz_file, pkg_xz_file]

        md5_path = path / "by-hash" / "MD5Sum"
        md5_path.mkdir(parents=True, exist_ok=True)
        sha1_path = path / "by-hash" / "SHA1"
        sha1_path.mkdir(parents=True, exist_ok=True)
        sha256_path = path / "by-hash" / "SHA256"
        sha256_path.mkdir(parents=True, exist_ok=True)
        for file in files:
            shutil.copy(file.path, md5_path / file.md5)
            shutil.copy(file.path, sha1_path / file.sha1)
            shutil.copy(file.path, sha256_path / file.sha256)

        return files
    
    def create_release(self):
        return [self.create_dist(self.path / "dists", dist) for dist in self.packages]
            
        
            
        

@dataclasses.dataclass
class KeyManager:
    homedir: pathlib.Path
    priv_key: bytes
    pub_key: bytes
    pub_key_name: str
    signer: str

    @classmethod
    def from_env(cls, path: pathlib.Path, *, enc_priv_key: str, enc_pub_key: str, signer: str, pub_key_name: str):
        priv_key = base64.b64decode(enc_priv_key)
        pub_key = base64.b64decode(enc_pub_key)
        return cls(path, priv_key, pub_key, pub_key_name, signer)

    def init(self):
        self.homedir.mkdir(parents=True, exist_ok=True, mode=0o700)
        subprocess.run(["gpg", "--import"], input=self.priv_key, check=True, env={"GNUPGHOME": str(self.homedir)}, capture_output=True)
    
    def create_sign_gpg(self, file: HashedFile, name: str = "Release.gpg"):
        p = subprocess.run(["gpg", "--default-key", self.signer, "-abs"], input=file.data, check=True, env={"GNUPGHOME": str(self.homedir)}, capture_output=True)
        file.path.with_name(name).write_bytes(p.stdout)
    
    def create_sign_clear(self, file: HashedFile, name: str = "InRelease"):
        p = subprocess.run(["gpg", "--default-key", self.signer, "-abs", "--clearsign"], input=file.data, check=True, env={"GNUPGHOME": str(self.homedir)}, capture_output=True)
        file.path.with_name(name).write_bytes(p.stdout)
    
    def save_pub_key_asc(self, path: pathlib.Path):
        (path / f"{self.pub_key_name}.asc").write_bytes(self.pub_key)
        
    def save_pub_key_gpg(self, path: pathlib.Path):
        subprocess.run(["gpg", "--dearmor", "-o", str((path / f"{self.pub_key_name}.gpg"))], input=self.pub_key, check=True, capture_output=True)
    
    def save_pub_key(self, path: pathlib.Path):
        self.save_pub_key_asc(path)
        self.save_pub_key_gpg(path)
        


def main():
    logging.basicConfig(level=logging.INFO)
    DEB_KEY_PUB = os.environ["DEB_KEY_PUB"]
    DEB_KEY_PRIV = os.environ["DEB_KEY_PRIV"]
    DEB_KEY_SIGNER = os.environ["DEB_KEY_SIGNER"]
    DEB_PUBLIC_KEY_NAME = os.environ["DEB_PUBLIC_KEY_NAME"]
    

    DEB_REPO_URL = os.environ["DEB_REPO_URL"]
    DEB_REPO_NAME = os.environ.get("DEB_REPO_NAME")

    GITHUB_WORKSPACE = pathlib.Path(os.environ.get("GITHUB_WORKSPACE", "."))
    PKGS_PATH = pathlib.Path(os.environ.get("PKGS_PATH", "pkgs"))
    BUILD_DIR = GITHUB_WORKSPACE / ".deb"
    REPO_DIR = BUILD_DIR / "deb"

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    REPO_DIR.mkdir(parents=True, exist_ok=True)

    keymgr = KeyManager.from_env(BUILD_DIR / "keys", enc_priv_key=DEB_KEY_PRIV, enc_pub_key=DEB_KEY_PUB, signer=DEB_KEY_SIGNER, pub_key_name=DEB_PUBLIC_KEY_NAME)
    keymgr.init()
    keymgr.save_pub_key(REPO_DIR)

    pkgs = find_packages(PKGS_PATH, DistComponent([Dist("stable")], Component("main")))
    
    repo = Repo(REPO_DIR)
    for p in pkgs:
        repo.add_binary(p)
    
    
    release_files = repo.create_release()

    for release_file in release_files:
        keymgr.create_sign_gpg(release_file, "Release.gpg")
        keymgr.create_sign_clear(release_file, "InRelease")


    HEADER = f"""	
```bash
# Install key
curl '{DEB_REPO_URL}/{DEB_PUBLIC_KEY_NAME}.asc' | sudo gpg --dearmor -o /usr/share/keyrings/{DEB_PUBLIC_KEY_NAME}.gpg
sudo curl -o /usr/share/keyrings/{DEB_PUBLIC_KEY_NAME}.gpg '{DEB_REPO_URL}/{DEB_PUBLIC_KEY_NAME}.gpg'
sudo curl -o /usr/share/keyrings/{DEB_PUBLIC_KEY_NAME}.asc '{DEB_REPO_URL}/{DEB_PUBLIC_KEY_NAME}.asc'

# Install repo
echo \"deb [arch=amd64 signed-by=/usr/share/keyrings/{DEB_PUBLIC_KEY_NAME}.gpg] {DEB_REPO_URL} stable main\" | sudo tee /etc/apt/sources.list.d/{DEB_REPO_NAME}.list" >> $@
```
# Debian Repository

"""

    create_static_indexes(REPO_DIR, HEADER)



if __name__ == "__main__":
    main()
