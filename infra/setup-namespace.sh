#!/bin/sh
set -eu
if ! temporal operator namespace describe --namespace default >/dev/null 2>&1; then
  temporal operator namespace create --namespace default --retention 168h
fi
