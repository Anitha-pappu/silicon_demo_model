#!/usr/bin/env bash

# Copyright (c) 2016 - present
# QuantumBlack Visual Analytics Ltd (a McKinsey company).
# All rights reserved.
#
# This software framework contains the confidential and proprietary information
# of QuantumBlack, its affiliates, and its licensors. Your use of these
# materials is governed by the terms of the Agreement between your organisation
# and QuantumBlack, and any unauthorised use is forbidden. Except as otherwise
# stated in the Agreement, this software framework is for your internal use
# only and may only be shared outside your organisation with the prior written
# permission of QuantumBlack.

# RUN FROM OPTIMUS REPO ROOT

# restrictive bash error handling
set -euo pipefail

# clean build cache and artifacts
rm -rf docs/build

# install sphinx dependencies
pip install -r ./docs/requirements.txt

# generate sphinx-apidoc rst files and land them in the out directory
# these are bits of python module documentation derived from doc strings
sphinx-apidoc -f --module-first -o docs/build/apidoc/optimizer src/optimizer
sphinx-apidoc -f --module-first -o docs/build/apidoc/modeling src/modeling
sphinx-apidoc -f --module-first -o docs/build/apidoc/feature_factory src/feature_factory
sphinx-apidoc -f --module-first -o docs/build/apidoc/preprocessing src/preprocessing
sphinx-apidoc -f --module-first -o docs/build/apidoc/recommend src/recommend
sphinx-apidoc -f --module-first -o docs/build/apidoc/reporting src/reporting

# sphinx-build -WEa --keep-going -j 2 -D language=en . ./docs/build/html
sphinx-build -Ea -j 2 -D language=en . ./docs/build/html -c ./docs
