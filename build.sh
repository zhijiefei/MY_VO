
echo "Configuring and building G2O ..."
cd Thirdparty/g2o
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

echo "Configuring and building DBOW2 ..."
cd ../../DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

echo "Configuring and building ORB_SLAM2 ..."
cd ../../../
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j


cd ../Vocabulary
echo "Converting vocabulary to binary version"
./bin_vocabulary
cd ..
