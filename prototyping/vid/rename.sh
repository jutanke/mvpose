moved_frame=0
for f in $(seq 0 5 335);
do
    fname_from=$(printf "frame%04d.png" $f)
    fname_to=$(printf "frame_new_%04d.png" $moved_frame)
    echo "$fname_from --> $fname_to"
    cp $fname_from $fname_to
    ((moved_frame++))

done
