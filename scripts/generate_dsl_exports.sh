SCRIPT=$(realpath $0)
DIRECTORY=$(cd `dirname $SCRIPT` &&  cd .. && pwd)
cd $DIRECTORY

#using venv if available
ACTIVATE="venv/bin/activate"
if [ -f $ACTIVATE ]; then
   echo "found activate"
   source $ACTIVATE
fi

echo $DIRECTORY

python --version
#python -m mlpipe.admin.generate_dsl_imports
cd -

