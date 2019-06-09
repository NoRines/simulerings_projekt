#!/bin/bash
while getopts 'd' flag; do
    case "${flag}" in
        d) # DEBUG FLAG
            ../../waf --run="project" --command-template="kdbg %s"
            exit 1
        ;;
    esac
done
../../waf --run="project"