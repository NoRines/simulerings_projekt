#!/bin/bash

for j in {1..400}
do
	../../waf --run="project --seed=$j"
done
