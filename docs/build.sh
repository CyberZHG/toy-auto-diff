#!/usr/bin/env bash
rm -r build
sphinx-apidoc -o ./source/ ../auto_diff/
sphinx-apidoc -o ./source/ ../demos/
make html
