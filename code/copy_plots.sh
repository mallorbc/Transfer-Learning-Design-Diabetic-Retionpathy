#!/bin/bash
copy_plots(){
for last; do true; done
	

array=( "$@" )
unset "array[${#array[@]}-1]"
final_loc=$last


current_dir=$(pwd)
for x in "${array[@]}";do
	folder=$current_dir
	folder+="/"
	folder+=$x
	folder+="/plots"
	dest=$final_loc
	dest+="/"
	dest+=$x

	if [ ! -d $dest ]; then
		mkdir -p $dest
	fi 
	cp -R $folder $dest

done
}
for last; do true; done
array=( "$@" )
unset "array[${#array[@]}-1]"


copy_plots $array $last


