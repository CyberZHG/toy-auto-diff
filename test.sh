#!/usr/bin/env bash
nosetests --with-coverage --cover-html --cover-html-dir=htmlcov --cover-package="auto_diff" tests
