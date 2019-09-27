#!/bin/bash

#Converts coord_last.xyz to coord_z.xyz for a new round of simulations, saving
#coord_previous.xyz in case you make a mistake

cp coord_z.xyz coord_previous.xyz

echo 'Current coord_z.xyz saved in coord_previous.xyz'

awk 'NR>2 {$1 = 1} {print}' coord_last.xyz > coord_z.xyz

echo 'Last frame from previous simulation changed into new input coord_z.xyz'

 