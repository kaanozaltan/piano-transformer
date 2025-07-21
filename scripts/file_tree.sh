#!/usr/bin/env bash

# Usage: ./file_tree.sh [folder] [threshold]
FOLDER="${1:-.}"
THRESHOLD="${2:-0}"

find "$FOLDER" -type f | awk -F/ -v threshold="$THRESHOLD" -v folder="$FOLDER" '
{
  path=""
  for (i=1; i<NF; i++) {
    path = (path ? path "/" : "") $i
    count[path]++
  }
}
END {
  root = folder
  if (root == ".") root = "."
  print count[root] " " root
  print_tree(root, "", 1)
}

function print_tree(dir, prefix, is_last,   p, n, i, j, tmp, child) {
  if (dir != folder && count[dir] >= threshold) {
    connector = (is_last ? "└── " : "├── ")
    print prefix connector count[dir] " " dir
    prefix = prefix (is_last ? "    " : "│   ")
  } else if (dir != "." && count[dir] < threshold) {
    return
  }

  n = 0
  for (p in count) {
    if (p ~ "^" dir "/[^/]+$" && count[p] >= threshold) {
      child[n++] = p
    }
  }

  for (i=0; i<n-1; i++)
    for (j=i+1; j<n; j++)
      if ((count[child[j]] > count[child[i]]) ||
          (count[child[j]] == count[child[i]] && child[j] < child[i])) {
        tmp=child[i]; child[i]=child[j]; child[j]=tmp
      }

  for (i=0; i<n; i++)
    print_tree(child[i], prefix, i==n-1)
}
'
