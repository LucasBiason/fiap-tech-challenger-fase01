#!/usr/bin/env bash

set -ef

cli_help() {
  cli_name=${0##*/}
  echo "
$cli_name
System entrypoint cli
Usage: $cli_name [command]
Commands:
  runserver     deploy runserver
  *             Help
"
  exit 1
}

case "$1" in
  train)
    python app/ml/wealth_cost_prediction.py
    ;;
  runserver)
    uvicorn app.main:app --host 0.0.0.0 --port 8004 --reload
    ;;
  *)
    cli_help
    ;;
esac
