#pragma once

#include "Config.h"
#include "DataSet.h"
#include "FactorGraph.h"
#include<math.h>
class EdgeFactorFunction: public FactorFunction
{
public:
	int     num_label;
	double* lambda;
	map<int, int>* feature_offset;

	EdgeFactorFunction(int num_label, double* p_lambda, map<int,int>* edge_feature_offset)
	{
		this->num_label = num_label;
		this->lambda = p_lambda;
		this->feature_offset = edge_feature_offset;
	}
	
	virtual double GetValue(int y1, int y2, int cat)
	{
		int a = y1 < y2 ? y1 : y2;
		int b = y1 > y2 ? y1 : y2;

		//several hardcoded potenial matrices based on edge type
		//if edge type isn't 0, 1, or 2, use the default learned potentials
		if(cat == 0){
			double pot0[4] = {0.515, 0.485, 0.485, 0.515};
			return pot0[a*num_label+b];
		}
		if(cat == 1){
			double pot1[4] = {0.485, 0.515, 0.515, 0.485};
			return pot1[a*num_label+b];
		}
		if(cat == 2){
			return 1-1e-5;
		}
		int i = (*feature_offset)[a * num_label + b];
		return exp (lambda[i]);
	}
};

class CRFModel
{
public:
	Config*     conf;
	DataSet*    train_data;
	DataSet*    test_data;

	int         num_sample;
	int         num_label;
	int         num_attrib_type;
	int         num_edge_type;
	
	int         num_feature;
	char	    *mapping;
	double      *lambda;
	//double	    *PyStateFactors; //for CNF
	FactorGraph *sample_factor_graph;

	int             num_attrib_parameter;
	int             num_edge_feature_each_type;
	map<int, int>   edge_feature_offset;
	EdgeFactorFunction**   func_list;

	CRFModel()
	{
	}

	void InitTrain(Config* conf, DataSet* train_data);
	void GenFeature();
	void SetupFactorGraphs();

	void Train();
	double CalcGradient(double* gradient);
	double CalcPartialLabeledGradientForSample(DataSample* sample, FactorGraph* factor_graph, double* gradient);

	void SelfEvaluate();
	void PartialSelfEvaluation();
	
	void InitEvaluate(Config* conf, DataSet* test_data);
	void Evalute();
	int GetAttribParameterId(int y, int x)
	{
		return y * num_attrib_type + x;
	}

	int GetEdgeParameterId(int edge_type, int a, int b)
	{ 
		int offset = edge_feature_offset[(a<b?a:b) * num_label + (a>b?a:b)];
		return num_attrib_parameter + edge_type * num_edge_feature_each_type + offset;
	}

    ~CRFModel()
	{
		Clean();
	}

	void Clean();

	void SaveModel(const char* file_name);
	void LoadModel(const char* file_name);

	void Estimate(Config* conf);
	void EstimateContinue(Config* conf);
	void Inference(Config* conf);

	int         N, M;
	vector<int> state;
	vector<int> best_state;
	vector<int> unlabeled;
	vector<int> test;
	vector<int> valid;

	void LoadStateFile(const char* filename);
	void MHTrain(Config *conf);
	double MHEvaluate(int max_iter = 0, bool isfinal = false);
	void SavePred(const char* filename);

	void MHTrain1(Config *conf);
	bool train1_sample(int type, int center, int ynew, double p, vector<int>& _state, map<int,double>& _gradient);
};
extern "C"{
void train(void* model);
void GetNNOutput(void* model, double* pyPtr, int dimension, double* logits, int dimension2);
void BackpropToNN(void* model, double* pyPtr, int dimension);
}
