# Maintainer: Mikhail Hushchyn
# OS: SCL, CentOS

echo "export SHIPSOFT=/afs/cern.ch/work/m/mhushchy/ShipSoft" >> ~/.bashrc
echo "export SIMPATH=$SHIPSOFT/FairSoftInst" >> ~/.bashrc
echo "export FAIRROOTPATH=$SHIPSOFT/FairRootInst" >> ~/.bashrc
echo "export FAIRSHIP=$SHIPSOFT/FairShip" >> ~/.bashrc
echo "export FAIRSHIPRUN=$SHIPSOFT/FairShipRun" >> ~/.bashrc


yum -y install cmake gcc gcc-c++ gcc-gfortran make patch sed \
  libX11-devel libXft-devel libXpm-devel libXext-devel \
  libXmu-devel mesa-libGLU-devel mesa-libGL-devel ncurses-devel \
  curl curl-devel bzip2 bzip2-devel gzip unzip tar \
  expat-devel subversion git flex bison imake redhat-lsb-core python-devel \
  libxml2-devel wget openssl-devel krb5-devel \
  automake autoconf libtool which; yum clean all

# Try for Docker
#RUN curl -L https://gitlab.cern.ch/linuxsupport/locmap/raw/master/potd7-stable.repo -o /etc/yum.repos.d/potd7-stable.repo
#RUN yum -y install locmap
#RUN yum -y install eos-fuse; yum clean all


# Install FairSoft
mkdir $SHIPSOFT
cd $SHIPSOFT
git clone https://github.com/ShipSoft/FairSoft.git
cd FairSoft
echo 1 | ./configure.sh

rm -rf $SHIPSOFT/FairSoft


# Install FairRoot
cd $SHIPSOFT
git clone  https://github.com/ShipSoft/FairRoot.git
cd FairRoot
mkdir build
./configure.sh

rm -rf $SHIPSOFT/FairRoot


# Install the SHIP software
cd $SHIPSOFT
git clone https://github.com/ShipSoft/FairShip.git
cd FairShip
./configure.sh

# Set up environment
source $SHIPSOFT/FairShipRun/config.sh


