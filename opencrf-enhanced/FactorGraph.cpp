#include "FactorGraph.h"
#include "Constant.h"
#include<time.h>
#include<string.h>
/**************************************************
 Node
**************************************************/

void Node::BasicInit(int num_label)
{
	this->num_label = num_label;
	msg = new double[num_label];
}

void VariableNode::Init(int num_label)
{
	BasicInit(num_label);

	state_factor = MatrixUtil::GetDoubleArr(num_label);

	marginal = MatrixUtil::GetDoubleArr(num_label);
}

void FactorNode::Init(int num_label)
{
	BasicInit(num_label);

	marginal = new double*[num_label];
	for (int i = 0; i < num_label; i++)
		marginal[i] = new double[num_label];
}

void Node::NormalizeMessage()
{
	double s = 0.0;
	for (int y = 0; y < num_label; y++)
		s += msg[y];
	for (int y = 0; y < num_label; y++)
		msg[y] /= s;
}

void VariableNode::BeliefPropagation(double* diff_max, bool labeled_given)
{
	double product;
	for (int i = 0; i < neighbor.size(); i++)
	{
		FactorNode* f = (FactorNode*) neighbor[i];
		for (int y = 0; y < num_label; y++)
		{
			product = this->state_factor[y];
			for (int j = 0; j < neighbor.size(); j++)
				if (i != j)
					product *= this->belief[j][y];
			msg[y] = product;
		}

		NormalizeMessage();
		f->GetMessageFrom(id, msg, diff_max);
	}
}

void FactorNode::BeliefPropagation(double* diff_max, bool labeled_given)
{
	//a tuple of the names of the two neighboring nodes
	//we have both directions here <neighbor[0],neighbor[1]> and <neighbor[1],neighbor[0]>
	std::tuple<std::string,std::string> edgePair0 = std::tuple<std::string,std::string>(((VariableNode*)neighbor[0])->nodename, ((VariableNode*)neighbor[1])->nodename);
	std::tuple<std::string,std::string> edgePair1 = std::tuple<std::string,std::string>(((VariableNode*)neighbor[1])->nodename, ((VariableNode*)neighbor[0])->nodename);
	for (int i = 0; i < 2; i++)
	{
		if (labeled_given && ((VariableNode*)neighbor[i])->label_type == Enum::KNOWN_LABEL)
		{
			for (int y = 0; y < num_label; y++)
				msg[y] = 0;
			msg[((VariableNode*)neighbor[i])->y] = 1.0;
		}
		else
		{
			int cat = -1; //default
			//look up edge category in map and extract edge type
			std::map<std::tuple<std::string,std::string>,int>::iterator it = this->potentialMapPtr->find(edgePair0);
			if(it != this->potentialMapPtr->end()) {
				cat = it->second;
			}
			//if we can't find it, look for the edge type of the other edge pair
			else{
				it = this->potentialMapPtr->find(edgePair1);
				if(it != this->potentialMapPtr->end()){
			       		cat = it->second;
				}
				else{
					fprintf(stderr, "something went wrong s s\n");//, edgePair0[0] ,edgePair0[1]);
				}
			}
			for (int y = 0; y < num_label; y++)
			{
				double s = 0;
				for (int y1 = 0; y1 < num_label; y1++)
					//pass edge type to GetValue
					s += func->GetValue(y, y1, cat) * belief[1 - i][y1];
				msg[y] = s;
			}
			NormalizeMessage();
		}

		neighbor[i]->GetMessageFrom(id, msg, diff_max);
	}
}

void VariableNode::MaxSumPropagation(double* diff_max, bool labeled_given)
{
	double product;

	for (int i = 0; i < neighbor.size(); i++)
	{
		FactorNode* f = (FactorNode*) neighbor[i];
		for (int y = 0; y < num_label; y++)        
		{
			product = this->state_factor[y];
			for (int j = 0; j < neighbor.size(); j++)
				if (i != j)
					product *= this->belief[j][y];
			msg[y] = product;
		}
		NormalizeMessage();
		f->GetMessageFrom(id, msg, diff_max);
	}
}

void FactorNode::MaxSumPropagation(double* diff_max, bool labeled_given)
{
	//pairs of node names of this factor node's neighbors
	std::tuple<std::string,std::string> edgePair0 = std::tuple<std::string,std::string>(((VariableNode*)neighbor[0])->nodename, ((VariableNode*)neighbor[1])->nodename);
	std::tuple<std::string,std::string> edgePair1 = std::tuple<std::string,std::string>(((VariableNode*)neighbor[1])->nodename, ((VariableNode*)neighbor[0])->nodename);
	for (int i = 0; i < 2; i++)    
	{
		if (labeled_given && ((VariableNode*)neighbor[i])->label_type == Enum::KNOWN_LABEL)
		{
			for (int y = 0; y < num_label; y++)
				msg[y] = 0;
			msg[((VariableNode*)neighbor[i])->y] = 1.0;
		}
		else
		{
			//like above, finding hte edge category
			int cat = -1;
			std::map<std::tuple<std::string,std::string>,int>::iterator it = this->potentialMapPtr->find(edgePair0);
			if(it != this->potentialMapPtr->end()) cat = it->second;
			else cat = this->potentialMapPtr->find(edgePair1)->second;
			for (int y = 0; y < num_label; y++)
			{
				double s = func->GetValue(y, 0, cat) * belief[1 - i][0];
				double tmp;
				for (int y1 = 0; y1 < num_label; y1 ++)
				{
					tmp = func->GetValue(y, y1, cat) * belief[1 - i][y1];
					if (tmp > s) s = tmp;
				}
				msg[y] = s;
			}
			NormalizeMessage();
		}

		neighbor[i]->GetMessageFrom(id, msg, diff_max);
	}
}


/**************************************************
 FactorGraph
**************************************************/

void FactorGraph::InitGraph(int n, int m, int num_label)
{
	this->labeled_given = false;
	this->n = n;
	this->m = m;
	this->num_label = num_label;
	this->num_node = n + m;
	
	var_node = new VariableNode[n];
	factor_node = new FactorNode[m];

	int p_node_id = 0;

	p_node = new Node*[n + m];
	/*
	//contains name of node name listing file for urren fold
	FILE* mapfile = fopen("../rawdata/mapfile.txt","r");
	fseek(mapfile,0,SEEK_END);
	long fs2 = ftell(mapfile);
	rewind(mapfile);
	char* mapname = mappin(char*)malloc(fs2+1);
	fread(mapname,1,fs2,mapfile);
	mapname[fs2-1] = '\0';
	fclose(mapfile);
	fprintf(stderr, "%s\n", mapname);
	*/
	//actual node name file for current fold
	FILE* fp = fopen(this->mapping,"r"); //e.g. "../rawdata/maps/map0.txt"
	fseek(fp,0,SEEK_END);
	long fs = ftell(fp);
	rewind(fp);
	char* buffer = new char[fs+1];
	fread(buffer,1,fs,fp);
	fclose(fp);
	//separate string into separate chunks
	for(int i = 0; i < fs; i++){
		if(buffer[i] == '\n') buffer[i] = '\0';
	}
	//create mapping of node name pairs to edge type
	this->potentialMap = std::map<std::tuple<std::string,std::string>,int>();

	//actual mapping of node pairs to edge types
	fp = fopen("/home/jakob/experiment/rawdata/edgecats.txt","r");
	fseek(fp,0,SEEK_END);
	fs = ftell(fp);
	rewind(fp);
	char* mapBuf = new char[fs+1];
	fread(mapBuf,1,fs,fp);
	fclose(fp);
	//fprintf(stderr, "%s\n",mapBuf); exit(0);
	//break string into chunks
	for(int i = 0; i < fs; i++){
		if(mapBuf[i] == '\n' || (mapBuf[i] == ' ' && i && mapBuf[i-1] != ',')) mapBuf[i] = '\0';
	}
	for (int i = 0; i < n; i++)
	{
		var_node[i].nodename = std::string(buffer); //each variable node gets a name from the data file
		buffer += strlen(buffer)+1;
		var_node[i].id = p_node_id;
		p_node[p_node_id ++] = &var_node[i];
		
		var_node[i].Init( num_label );
	}
	for (int i = 0; i < m; i++)
	{
		factor_node[i].id = p_node_id;
		p_node[p_node_id ++] = &factor_node[i];

		factor_node[i].Init( num_label );
		factor_node[i].potentialMapPtr = &potentialMap;

		//for each edge, get the names of the two nodes involved, and the edge type
		char* id1 = mapBuf;
		mapBuf += strlen(mapBuf)+1;
		char* id2 = mapBuf;
		mapBuf += strlen(mapBuf)+1;
		int etype = atoi(mapBuf);
		//put mapping into potentialMap
		potentialMap[std::tuple<std::string,std::string>(std::string(id1),std::string(id2))] = etype;
		mapBuf += strlen(mapBuf)+1;
	} 
	factor_node_used = 0;
}

void FactorGraph::AddEdge(int a, int b, FactorFunction* func)
{
	// AddEdge can be called at most m times
	if (factor_node_used == m) return;

	factor_node[factor_node_used].func = func;
	
	factor_node[factor_node_used].AddNeighbor( &var_node[a] );
	factor_node[factor_node_used].AddNeighbor( &var_node[b] );

	var_node[a].AddNeighbor( &factor_node[factor_node_used] );
	var_node[b].AddNeighbor( &factor_node[factor_node_used] );

	factor_node_used++;
}

void FactorGraph::ClearDataForSumProduct()
{   
	for (int i = 0; i < n; i++)
	{
		MatrixUtil::DoubleArrFill(var_node[i].state_factor, num_label, 1.0 / num_label);          
	}

	for (int i = 0; i < num_node; i++)
	{
		for (int t = 0; t < p_node[i]->neighbor.size(); t++)
		{
			MatrixUtil::DoubleArrFill(p_node[i]->belief[t], num_label, 1.0 / num_label);
		}
	}
}

void FactorGraph::ClearDataForMaxSum()
{
	for (int i = 0; i < n; i++)
	{
		MatrixUtil::DoubleArrFill(var_node[i].state_factor, num_label, 1.0 / num_label);
	}
	for (int i = 0; i < num_node; i++)
	{
		for (int t = 0; t < p_node[i]->neighbor.size(); t++)
		{
			for (int y = 0; y < num_label; y++)
				p_node[i]->belief[t][y] = 1.0 / num_label;
		}
	}
}

void FactorGraph::GenPropagateOrder()
{
	bool* mark = new bool[num_node];
	bfs_node = new Node*[num_node];

	for (int i = 0; i < num_node; i++)
		mark[i] = false;

	int head = 0, tail = -1;
	for (int i = 0; i < num_node; i++)
	{
		if (! mark[i])
		{
			entry.push_back( p_node[i] );
			bfs_node[++tail] = p_node[i];
			mark[p_node[i]->id] = 1;

			while (head <= tail)
			{
				Node* u = bfs_node[head++];
				for (vector<Node*>::iterator it = u->neighbor.begin(); it != u->neighbor.end(); it++)
					if (! mark[(*it)->id] )
					{
						bfs_node[++tail] = *it;
						mark[(*it)->id] = 1;
					}
			}
		}
	}

	delete[] mark;
}

void FactorGraph::BeliefPropagation(int max_iter)
{    
	int start, end, dir;

	converged = false;
	for (int iter = 0; iter < max_iter; iter++)
	{
		diff_max = 0.0;

		if (iter % 2 == 0)
			start = num_node - 1, end = -1, dir = -1;
		else
			start = 0, end = num_node, dir = +1;
		
		for (int p = start; p != end; p += dir)
		{
			bfs_node[p]->BeliefPropagation(&diff_max, this->labeled_given);
		}

		if (diff_max < 1e-6) break;
	}
}

void FactorGraph::CalculateMarginal()
{
	FILE* fp = fopen("marginals.txt","w+");
	for (int i = 0; i < n; i++)
	{
		double sum_py = 0.0;
		for (int y = 0; y < num_label; y++){
			var_node[i].marginal[y] = var_node[i].state_factor[y];
			for (int t = 0; t < var_node[i].neighbor.size(); t++)
				var_node[i].marginal[y] *= var_node[i].belief[t][y];
			sum_py += var_node[i].marginal[y];
		}
		for (int y = 0; y < num_label; y++)
		{
			var_node[i].marginal[y] /= sum_py;
			fprintf(fp, "%f ", var_node[i].marginal[y]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
	fprintf(stderr, "lol");
	for (int i = 0; i < m; i++)
	{
		double sump = 0.0;
		//again, getting the edge type for the pair of nodes associated with this edge
		std::tuple<std::string,std::string> edgePair0 = std::tuple<std::string,std::string>(((VariableNode*)(factor_node[i].neighbor[0]))->nodename, ((VariableNode*)(factor_node[i].neighbor[1]))->nodename);
		std::tuple<std::string,std::string> edgePair1 = std::tuple<std::string,std::string>(((VariableNode*)(factor_node[i].neighbor[1]))->nodename, ((VariableNode*)(factor_node[i].neighbor[0]))->nodename);
		int cat = -1;
		std::map<std::tuple<std::string,std::string>,int>::iterator it = this->potentialMap.find(edgePair0);
		if(it != this->potentialMap.end()) cat = it->second;
		else cat = this->potentialMap.find(edgePair1)->second;
		for (int a = 0; a < num_label; a++)
			for (int b = 0; b < num_label; b++)
			{
				factor_node[i].marginal[a][b] +=
					factor_node[i].belief[0][a]
					* factor_node[i].belief[1][b]
					* factor_node[i].func->GetValue(a, b, cat);
				sump += factor_node[i].marginal[a][b];
			}
		for (int a = 0; a < num_label; a++)
			for (int b = 0; b < num_label; b++)
				factor_node[i].marginal[a][b] /= sump;
	}
}

void FactorGraph::MaxSumPropagation(int max_iter)
{
	int start, end, dir;

	converged = false;
	for (int iter = 0; iter < max_iter; iter++)
	{
		diff_max = 0;

		if (iter % 2 == 0)
			start = num_node - 1, end = -1, dir = -1;
		else
			start = 0, end = num_node, dir = +1;

		for (int p = start; p != end; p += dir)
		{
			bfs_node[p]->MaxSumPropagation(&diff_max, labeled_given);
		}

		if (diff_max < 1e-6) break;
	}
}

void FactorGraph::Clean()
{
	if (var_node) delete[] var_node;
	if (factor_node) delete[] factor_node;
	if (p_node) delete[] p_node;
	if (bfs_node) delete[] bfs_node;
}
