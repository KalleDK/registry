ifndef GITHUB_WORKSPACE
GITHUB_WORKSPACE := .
endif

ifndef PKGS_PATH
PKGS_PATH := deb
endif

BUILD_DIR := $(GITHUB_WORKSPACE)/.deb
REPO_DIR := $(BUILD_DIR)/repo
REPO_POOL := $(REPO_DIR)/pool/main
DIST := stable
DIST_REL := dists/$(DIST)
DIST_DIR := $(REPO_DIR)/$(DIST_REL)

KEY_DIR := $(BUILD_DIR)/keys

SOURCE_DEBS = $(wildcard $(PKGS_PATH)/*.deb)
REPO_DEBS = $(patsubst  $(PKGS_PATH)/%, $(REPO_POOL)/%, $(SOURCE_DEBS))


AMD64_BINARY := $(DIST_DIR)/main/binary-amd64
AMD64_PACKAGES := $(AMD64_BINARY)/Packages
ARM64_BINARY := $(DIST_DIR)/main/binary-arm64
ARM64_PACKAGES := $(AMD64_BINARY)/Packages
I386_BINARY := $(DIST_DIR)/main/binary-i386
I386_PACKAGES := $(I386_BINARY)/Packages

HAS_AMD != dpkg-scanpackages --arch amd64 $(PKGS_PATH) | wc -l
HAS_ARM64 != dpkg-scanpackages --arch arm64 $(PKGS_PATH) | wc -l
HAS_I386 != dpkg-scanpackages --arch i386 $(PKGS_PATH) | wc -l

ifneq ($(HAS_ARM64), 0)
ARCHS += arm64
endif
ifneq ($(HAS_AMD), 0)
ARCHS += amd64
endif
ifneq ($(HAS_I386), 0)
ARCHS += i386
endif

ARCH_DIRS = $(ARCHS:%=$(DIST_DIR)/main/binary-%)
PACKAGES = $(ARCH_DIRS:%=%/Packages)
PACKAGES_GZ = $(PACKAGES:%=%.gz)

.PHONY: all
all: $(DIST_DIR)/Release.gpg $(DIST_DIR)/InRelease ${REPO_DIR}/index.md ${REPO_DIR}/${DEB_PUBLIC_KEY_NAME}.gpg ${REPO_DIR}/${DEB_PUBLIC_KEY_NAME}.asc
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

$(DIST_DIR): $(REPO_DIR)
	mkdir -p $@

$(REPO_POOL)/%.deb: $(PKGS_PATH)/%.deb $(REPO_POOL)
	cp "$<" "$@"

$(ARCH_DIRS): $(DIST_DIR)
	mkdir -p $@

$(PACKAGES): $(DIST_DIR)/main/binary-%/Packages: $(DIST_DIR)/main/binary-% $(REPO_DEBS)
	cd "$(REPO_DIR)" && dpkg-scanpackages --arch $* pool/ > $(DIST_REL)/main/binary-$*/Packages
	cd "$(REPO_DIR)" && dpkg-scanpackages --arch $* pool/ | grep Filename > $(DIST_REL)/main/binary-$*/index.md

$(PACKAGES_GZ): $(DIST_DIR)/main/binary-%/Packages.gz: $(DIST_DIR)/main/binary-%/Packages
	cat "$<" | gzip -9 > "$@"

$(DIST_DIR)/Release: $(PACKAGES) $(PACKAGES_GZ)
	cd $(DIST_DIR); \
	echo -n '' > Release; \
	echo "Origin: KalleDK" >> Release; \
	echo "Label: Debian" >> Release; \
	echo "Suite: $(DIST)" >> Release; \
	echo "Codename: bookworm" >> Release; \
	echo "Version: 12.2" >> Release; \
	echo "Architectures: $(ARCHS)" >> Release; \
	echo "Components: main" >> Release; \
	echo "Description: Repository for KalleDK releases" >> Release; \
	echo "Date: $$(date -Ru)" >> Release; \
	
	cd $(DIST_DIR); \
	echo "MD5Sum:" >> Release; \
	for arch in $(ARCHS); do \
		for f in main/binary-$${arch}/Packages.gz main/binary-$${arch}/Packages; do \
			echo " $$(md5sum $$f | cut -d" " -f1) $$(wc -c $$f)" >> Release; \
		done; \
	done
	
	cd $(DIST_DIR); \
	echo "SHA1:" >> Release; \
	for arch in $(ARCHS); do \
		for f in main/binary-$${arch}/Packages.gz main/binary-$${arch}/Packages; do \
			echo " $$(sha1sum $$f | cut -d" " -f1) $$(wc -c $$f)" >> Release; \
		done; \
	done
	cd $(DIST_DIR); \
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

$(DIST_DIR)/Release.gpg: $(DIST_DIR)/Release $(KEY_DIR)/trustdb.gpg
	cat $< | GNUPGHOME=$(KEY_DIR) gpg --default-key "$(DEB_KEY_SIGNER)" -abs > $@

$(DIST_DIR)/InRelease: $(DIST_DIR)/Release $(KEY_DIR)/trustdb.gpg
	cat $< | GNUPGHOME=$(KEY_DIR) gpg --default-key "$(DEB_KEY_SIGNER)" -abs --clearsign > $@

${REPO_DIR}/${DEB_PUBLIC_KEY_NAME}.asc: ${REPO_DIR}
	printf -- "$(DEB_KEY_PUB)" | base64 -d > $@

${REPO_DIR}/${DEB_PUBLIC_KEY_NAME}.gpg: ${REPO_DIR}/${DEB_PUBLIC_KEY_NAME}.asc
	cat $< | gpg --dearmor -o $@

${REPO_DIR}/index.md: ${REPO_DIR}
	echo -n '' > $@
	echo "# Debian Repository" >> $@
	echo "" >> $@
	echo "\`\`\`bash" >> $@
	echo "# Install key" >> $@
	echo "curl '${DEB_REPO_URL}/${DEB_PUBLIC_KEY_NAME}.asc' | sudo gpg --dearmor -o /usr/share/keyrings/${DEB_PUBLIC_KEY_NAME}.gpg" >> $@
	echo "sudo curl -o /usr/share/keyrings/${DEB_PUBLIC_KEY_NAME}.gpg '${DEB_REPO_URL}/${DEB_PUBLIC_KEY_NAME}.gpg'" >> $@
	echo "sudo curl -o /usr/share/keyrings/${DEB_PUBLIC_KEY_NAME}.asc '${DEB_REPO_URL}/${DEB_PUBLIC_KEY_NAME}.asc'" >> $@
	echo "" >> $@
	echo "# Install repo" >> $@
	echo "echo \"deb [arch=amd64 signed-by=/usr/share/keyrings/${DEB_PUBLIC_KEY_NAME}.gpg] ${DEB_REPO_URL} $(DIST) main\" | sudo tee /etc/apt/sources.list.d/${DEB_REPO_NAME}.list" >> $@
	echo "" >> $@
	echo "\`\`\`" >> $@