#!/bin/bash

EOS_DIR="$1"

if [ -z "$EOS_DIR" ]; then
  echo "Usage: $0 /store/user/.../directory"
  exit 1
fi

echo "Deleting files < 1GB in $EOS_DIR..."

eosls -l "$EOS_DIR" | awk -v dir="$EOS_DIR" '
{
  size = $5
  name = $NF

  if (size < 1000000000) {
    cmd = "eosrm " dir "/" name
    print cmd
    system(cmd)
  }
}
'
