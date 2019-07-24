#include "Config.h"
#include "DataSet.h"
#include "CRFModel.h"

extern "C"{
void* InitializeCRF(char* gradientstep, char* trainfile, char* srcmodel, char* dstmodel, char* mapping, int dim)
{
	int argc = 14;
	char* argv[argc];// = (char**)malloc(argc*sizeof(char*));
	argv[1] = "-estc";
	argv[2] = "-gradientstep";
	argv[3] = gradientstep;
	argv[4] = "-trainfile";
	argv[5] = trainfile;
	argv[6] = "-srcmodel";
	argv[7] = srcmodel;
	argv[8] = "-dstmodel";
	argv[9] = dstmodel;
	argv[10] = "-mapping";
	argv[11] = mapping;
	argv[12] = "-niter";
	argv[13] = "1";
	// Load Configuartion
	Config* conf = new Config();
	if (! conf->LoadConfig(argc, argv))
	{
		conf->ShowUsage();
		exit( 0 );
	}

	CRFModel *model = new CRFModel();
	if (conf->task == "-est")
	{
		model->Estimate(conf);
	}
	else if (conf->task == "-estc")
	{
		model->EstimateContinue(conf);
	}
	else if (conf->task == "-inf")
	{
		model->Inference(conf);
		
	}
	else
	{
		Config::ShowUsage();
	}
	return (void*)model;
	//return 0;
}}
