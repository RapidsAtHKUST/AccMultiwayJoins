#!/bin/bash

find -name \*parallel\*.conf -delete
find -name \*independent\*.conf -delete
find -name \*radix\*.conf -delete
find -maxdepth 1 -type d -empty -delete
