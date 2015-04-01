apt-get --assume-yes install wget
apt-get --assume-yes install git
apt-get --assume-yes install maven
apt-get --assume-yes install default-jdk
apt-get --assume-yes install g++
apt-get --assume-yes install python-dev
apt-get --assume-yes install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
apt-get --assume-yes install gfortran
wget http://www.netlib.org/blas/blas.tgz
gunzip blas.tgz
tar -xf blas.tar
rm blas.tgz
rm blas.tar
cd BLAS
gfortran -c *.f
ar rv libblas.a *.o
su -c "cp libblas.a /usr/local/lib"
apt-get --assume-yes install python-pip
pip install --upgrade --no-deps git+https://github.com/Theano/Theano.git
wget http://www.gtlib.gatech.edu/pub/apache/spark/spark-1.3.0/spark-1.3.0.tgz
gunzip spark-1.3.0.tgz
tar -xf spark-1.3.0.tar 
rm spark-1.3.0.tgz
rm spark-1.3.0.tar  
