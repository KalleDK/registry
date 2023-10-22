ifndef GITHUB_WORKSPACE
GITHUB_WORKSPACE := .
endif

ifndef PKGS_PATH
PKGS_PATH := deb
endif

BUILD_DIR := $(GITHUB_WORKSPACE)/.deb
REPO_DIR := $(BUILD_DIR)/repo
REPO_POOL := $(REPO_DIR)/pool/main
STABLE_DIR := $(REPO_DIR)/dists/stable

KEY_DIR := $(BUILD_DIR)/keys

SOURCE_DEBS = $(wildcard $(PKGS_PATH)/*.deb)
REPO_DEBS = $(patsubst  $(PKGS_PATH)/%, $(REPO_POOL)/%, $(SOURCE_DEBS))


AMD64_BINARY := $(STABLE_DIR)/main/binary-amd64
AMD64_PACKAGES := $(AMD64_BINARY)/Packages
ARM64_BINARY := $(STABLE_DIR)/main/binary-arm64
ARM64_PACKAGES := $(AMD64_BINARY)/Packages
I386_BINARY := $(STABLE_DIR)/main/binary-i386
I386_PACKAGES := $(I386_BINARY)/Packages

HAS_AMD != dpkg-scanpackages --arch amd64 deb | wc -l
HAS_ARM64 != dpkg-scanpackages --arch arm64 deb | wc -l
HAS_I386 != dpkg-scanpackages --arch i386 deb | wc -l

ifneq ($(HAS_ARM64), 0)
ARCHS += arm64
endif
ifneq ($(HAS_AMD), 0)
ARCHS += amd64
endif
ifneq ($(HAS_I386), 0)
ARCHS += i386
endif

ARCH_DIRS = $(ARCHS:%=$(STABLE_DIR)/main/binary-%)
PACKAGES = $(ARCH_DIRS:%=%/Packages)
PACKAGES_GZ = $(PACKAGES:%=%.gz)

.PHONY: all
all: $(STABLE_DIR)/Release.gpg $(STABLE_DIR)/InRelease
	echo $(PACKAGES_GZ)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

$(BUILD_DIR):
	mkdir -p $@

$(REPO_DIR):
	mkdir -p $@

$(KEY_DIR): $(BUILD_DIR)
	mkdir -p $@

$(REPO_POOL): $(REPO_DIR)
	mkdir -p $@

$(STABLE_DIR): $(REPO_DIR)
	mkdir -p $@



$(REPO_POOL)/%.deb: $(PKGS_PATH)/%.deb $(REPO_POOL)
	cp "$<" "$@"

$(ARCH_DIRS): $(STABLE_DIR)
	mkdir -p $@

$(PACKAGES): $(STABLE_DIR)/main/binary-%/Packages: $(STABLE_DIR)/main/binary-% $(REPO_DEBS)
	cd "$(REPO_DIR)" && dpkg-scanpackages --arch $* pool/ > dists/stable/main/binary-$*/Packages

$(PACKAGES_GZ): $(STABLE_DIR)/main/binary-%/Packages.gz: $(STABLE_DIR)/main/binary-%/Packages
	cat "$<" | gzip -9 > "$@"


$(STABLE_DIR)/Release: $(PACKAGES) $(PACKAGES_GZ)
	cd $(STABLE_DIR); \
	echo -n '' > Release; \
	echo "Origin: KalleDK" >> Release; \
	echo "Label: Debian" >> Release; \
	echo "Suite: stable" >> Release; \
	echo "Codename: bookworm" >> Release; \
	echo "Version: 12.2" >> Release; \
	echo "Architectures: $(ARCHS)" >> Release; \
	echo "Components: main" >> Release; \
	echo "Description: Repository for KalleDK releases" >> Release; \
	echo "Date: $$(date -Ru)" >> Release; \
	
	cd $(STABLE_DIR); \
	echo "MD5Sum:" >> Release; \
	for arch in $(ARCHS); do \
		for f in main/binary-$${arch}/Packages.gz main/binary-$${arch}/Packages; do \
			echo " $$(md5sum $$f | cut -d" " -f1) $$(wc -c $$f)" >> Release; \
		done; \
	done
	
	cd $(STABLE_DIR); \
	echo "SHA1:" >> Release; \
	for arch in $(ARCHS); do \
		for f in main/binary-$${arch}/Packages.gz main/binary-$${arch}/Packages; do \
			echo " $$(sha1sum $$f | cut -d" " -f1) $$(wc -c $$f)" >> Release; \
		done; \
	done
	cd $(STABLE_DIR); \
	echo "SHA256:" >> Release; \
	for arch in $(ARCHS); do \
		for f in main/binary-$${arch}/Packages.gz main/binary-$${arch}/Packages; do \
			echo " $$(sha256sum $$f | cut -d" " -f1) $$(wc -c $$f)" >> Release; \
		done; \
	done

$(KEY_DIR)/trustdb.gpg: $(KEY_DIR)
	echo $@
	GNUPGHOME=$(KEY_DIR) gpg --list-keys
	printf -- "$(DEB_KEY_PRIV)" | base64 -d | GNUPGHOME=$(KEY_DIR) gpg --import

$(STABLE_DIR)/Release.gpg: $(STABLE_DIR)/Release $(KEY_DIR)/trustdb.gpg
	cat $< | GNUPGHOME=$(KEY_DIR) gpg --default-key "$(DEB_KEY_SIGNER)" -abs > $@

$(STABLE_DIR)/InRelease: $(STABLE_DIR)/Release $(KEY_DIR)/trustdb.gpg
	cat $< | GNUPGHOME=$(KEY_DIR) gpg --default-key "$(DEB_KEY_SIGNER)" -abs --clearsign > $@
