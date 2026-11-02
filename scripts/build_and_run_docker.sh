#!/bin/bash
set -e

docker build -t credit-default-api .
docker run --rm -p 8000:8000 credit-default-api
