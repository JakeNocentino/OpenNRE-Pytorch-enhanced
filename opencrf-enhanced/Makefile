main: CRFModel.o Config.o DataSet.o FactorGraph.o MHTrain.o MHTrain1.o Util.o OpenCRF_Main.o
	g++ -g -o main *.o 

shared: CRFModel.o Config.o DataSet.o FactorGraph.o MHTrain.o MHTrain1.o Util.o Init.o
	g++ *.o -shared -o crflib.so

CRFModel.o: CRFModel.cpp
	g++ -g -c -fPIC CRFModel.cpp

Config.o: Config.cpp
	g++ -g -c -fPIC Config.cpp

DataSet.o: DataSet.cpp
	g++ -g -c -fPIC DataSet.cpp

FactorGraph.o: FactorGraph.cpp
	g++ -g -c -fPIC FactorGraph.cpp

MHTrain.o: MHTrain.cpp
	g++ -g -c -fPIC MHTrain.cpp

MHTrain1.o: MHTrain1.cpp
	g++ -g -c -fPIC MHTrain1.cpp

Util.o: Util.cpp
	g++ -g -c -fPIC Util.cpp

Init.o: Init.cpp
	g++ -c -fPIC Init.cpp

OpenCRF_Main.o: OpenCRF_Main.cpp
	g++ -g -c -fPIC OpenCRF_Main.cpp

clean: 
	rm main *.o

