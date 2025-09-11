# Bundle settings
DIST_DIR = dist
ROOT_DIR = ..
VERSION_MAJOR := $(shell grep "SPARSEIR_VERSION_MAJOR" $(ROOT_DIR)/include/sparseir/version.h | awk '{print $$3}')
VERSION_MINOR := $(shell grep "SPARSEIR_VERSION_MINOR" $(ROOT_DIR)/include/sparseir/version.h | awk '{print $$3}')
VERSION_PATCH := $(shell grep "SPARSEIR_VERSION_PATCH" $(ROOT_DIR)/include/sparseir/version.h | awk '{print $$3}')
PACKAGE_NAME = libsparseir-$(VERSION_MAJOR).$(VERSION_MINOR).$(VERSION_PATCH)

# Directories
SRC_DIR = src
INCLUDE_DIR = include
FORTRAN_DIR = fortran
DEPS_DIR = deps

# Include paths
#INCLUDES = -I$(INCLUDE_DIR) -I$(DEPS_DIR)/eigen3 -I$(DEPS_DIR)/xprec/include

.PHONY: all clean dist prepare deps

all: dist

prepare:
	@mkdir -p $(SRC_DIR) $(INCLUDE_DIR) $(FORTRAN_DIR)
	cp -r $(ROOT_DIR)/src/* $(SRC_DIR)/
	cp -r $(ROOT_DIR)/include/* $(INCLUDE_DIR)/
	cp -r $(ROOT_DIR)/fortran/* $(FORTRAN_DIR)/

deps:
	@mkdir -p $(DEPS_DIR)
	@if [ ! -d "$(DEPS_DIR)/eigen3" ]; then \
		tar xzf eigen3.tar.gz -C $(DEPS_DIR) && \
		mv $(DEPS_DIR)/eigen-* $(DEPS_DIR)/eigen3; \
	fi
	@if [ ! -d "$(DEPS_DIR)/xprec" ]; then \
		tar xzf xprec.tar.gz -C $(DEPS_DIR) && \
		mv $(DEPS_DIR)/libxprec-* $(DEPS_DIR)/xprec; \
	fi

clean:
	rm -rf $(DIST_DIR) $(DEPS_DIR) $(SRC_DIR) $(INCLUDE_DIR) $(FORTRAN_DIR)

dist: prepare deps
	@mkdir -p $(DIST_DIR)
	@mkdir -p $(DIST_DIR)/$(PACKAGE_NAME)
	cp -r Makefile.bundle $(SRC_DIR) $(INCLUDE_DIR) $(FORTRAN_DIR) $(DEPS_DIR) $(DIST_DIR)/$(PACKAGE_NAME)/
	cd $(DIST_DIR)/$(PACKAGE_NAME) && mv Makefile.bundle Makefile
	cd $(DIST_DIR) && tar czf $(PACKAGE_NAME).tar.gz $(PACKAGE_NAME)
	#rm -rf $(DIST_DIR)/$(PACKAGE_NAME) 