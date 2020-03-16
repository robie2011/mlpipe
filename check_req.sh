while read line || [[ -n $line ]]; do
    lib="${line/%==*/}"
    egrep -r "(import|from) $lib.*" mlpipe >/dev/null
    if (( $? == 1 )); then
        echo $lib; 
    fi
    
done < requirements.txt