#!/usr/bin/env bash
rm -r build
sphinx-apidoc -o ./source/ ../auto_diff/
make html
