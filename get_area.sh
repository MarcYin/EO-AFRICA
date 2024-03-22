#!/bin/bash
year=$1


#x0=`awk < SF_${year}.geojson -F\" '{for(i=1;i<=NF;i++)if($i == "coordinates") print($(i+1))}' | awk '{print $3,$4}' | sed 's/,//g'| awk -v dx=.00988282531187 -v dy=.00988282531187 '{print $1+dx}'| sort -n  | tail  -1`
#y0=`awk < SF_${year}.geojson -F\" '{for(i=1;i<=NF;i++)if($i == "coordinates") print($(i+1))}' | awk '{print $3,$4}' | sed 's/,//g'| awk -v dx=.00988282531187 -v dy=.00988282531187 '{print $2+dy}'| sort -n  | tail  -1`
#x1=`awk < SF_${year}.geojson -F\" '{for(i=1;i<=NF;i++)if($i == "coordinates") print($(i+1))}' | awk '{print $3,$4}' | sed 's/,//g'| awk -v dx=.00988282531187 -v dy=.00988282531187 '{print $1-dx}'| sort -n  | head  -1`
#y1=`awk < SF_${year}.geojson -F\" '{for(i=1;i<=NF;i++)if($i == "coordinates") print($(i+1))}' | awk '{print $3,$4}' | sed 's/,//g'| awk -v dx=.00988282531187 -v dy=.00988282531187 '{print $2-dy}'| sort -n  | head  -1`


x0=`awk < SF_${year}.geojson  -F, '($1*1==$1){printf("%f ",$1);getline;print($1)}'| awk -v dx=.00988282531187 -v dy=.00988282531187 '{print $1+dx}'| sort -n  | tail  -1`
y0=`awk < SF_${year}.geojson  -F, '($1*1==$1){printf("%f ",$1);getline;print($1)}'| awk -v dx=.00988282531187 -v dy=.00988282531187 '{print $2+dy}'| sort -n  | tail  -1`
x1=`awk < SF_${year}.geojson  -F, '($1*1==$1){printf("%f ",$1);getline;print($1)}'| awk -v dx=.00988282531187 -v dy=.00988282531187  '{print $1-dx}'| sort -n  | head  -1`
y1=`awk < SF_${year}.geojson  -F, '($1*1==$1){printf("%f ",$1);getline;print($1)}'| awk -v dx=.00988282531187 -v dy=.00988282531187 '{print $2-dy}'| sort -n  | head  -1`

sed < files/wider_area.geojson 's/XMAX/'${x0}'/g' | sed 's/YMAX/'${y0}'/g' | sed 's/XMIN/'${x1}'/g' | sed 's/YMIN/'${y1}'/g' > wider_area_${year}.geojson



