prefix=$(dirname -- "$0")
index="20 52 51"
nberr=0
for i in ${index}; do
  echo "Testing spaces_comm on index $i"
  $prefix/test_DDR_comm $i
  nberr=$(($nberr+$?))
done
for i in ${index}; do
  echo "Testing spaces_exact on index $i"
  $prefix/test_DDR_exact $i
  nberr=$(($nberr+$?))
done
for i in ${index}; do
  echo "Testing spaces_potential on index $i"
  $prefix/test_DDR_potential $i
  nberr=$(($nberr+$?))
done
for i in ${index}; do
  echo "Testing spaces_interpolate on index $i"
  $prefix/test_DDR_interpolate $i
  nberr=$(($nberr+$?))
done
for i in ${index}; do
  echo "Testing spaces_L2product on index $i"
  $prefix/test_DDR_L2product $i
  nberr=$(($nberr+$?))
done
echo "Found $nberr errors"

