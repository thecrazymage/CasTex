#!/bin/bash

#  -d,   --dir <path>       Target directory to evaluate (Required)
#  -ef,  --efiles <num>     Expected number of files in each folder (Required)
#  -ed,  --edirs <num>      Expected number of subdirectories in each folder (Required)
#  -esf, --esubfiles <num>  Expected number of files in each subdirectory (Required)

TXT_FILE="./objaverse_eval/assets/objaverse_subset.txt"
GLOBAL_ISSUE=0

while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--dir) DIR="$2"; shift 2 ;;
    -ef|--efiles) EFILES="$2"; shift 2 ;;
    -ed|--edirs) EDIRS="$2"; shift 2 ;;
    -esf|--esubfiles) ESUBFILES="$2"; shift 2 ;;
    *) shift ;;
  esac
done

if [[ -z "$DIR" || -z "$EFILES" || -z "$EDIRS" || -z "$ESUBFILES" ]]; then
    echo "Args missing. Usage: $0 -d <dir> -ef <num> -ed <num> -esf <num>"
    exit 1
fi

if [ -f "$TXT_FILE" ]; then
    while read -r line; do
        [[ -n "$line" && ! -d "$DIR/$line" ]] && echo "Missing folder: $line" && GLOBAL_ISSUE=1
    done < "$TXT_FILE"
fi

for folder in "$DIR"/*; do
    [ -d "$folder" ] || continue
    
    f_count=$(find "$folder" -maxdepth 1 -type f | wc -l)
    d_count=$(find "$folder" -maxdepth 1 -mindepth 1 -type d | wc -l)
    
    issue=0
    [[ $f_count -ne $EFILES ]] && issue=1
    [[ $d_count -ne $EDIRS ]] && issue=1

    if [ $d_count -gt 0 ]; then
        for sub in "$folder"/*/; do
            [ -d "$sub" ] || continue
            sub_f_count=$(find "$sub" -maxdepth 1 -type f | wc -l)
            [[ $sub_f_count -ne $ESUBFILES ]] && issue=1
        done
    fi

    [[ $issue -eq 1 ]] && basename "$folder" && GLOBAL_ISSUE=1
done

if [[ $GLOBAL_ISSUE -eq 0 ]]; then
    echo -e "\n\n\033[1;32m✓ Everything was rendered correctly! All checks passed!\033[0m\n\n"
else
    echo -e "\n\n\033[1;31m✗ Some checks failed: NOT everything was rendered correctly. Review output above.\033[0m\n\n"
fi