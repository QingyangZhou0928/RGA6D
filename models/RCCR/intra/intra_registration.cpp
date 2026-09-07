// ------------------------------------------------------------------------------------
// Modified from MAC (https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques)
// Originally authored by Xiyu Zhang (Copyright (c) 2023 zhangxy0517)
// Licensed under the MIT License.
// Modifications copyright (c) 2026 Qingyang Zhou
// ------------------------------------------------------------------------------------
#include <vector>
#include <utility>
#include <cstdio>
#include <cmath>
#include <unsupported/Eigen/MatrixFunctions>

#include <stdio.h>
#include <time.h>

#include <pcl/point_types.h>
#include <pcl/registration/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>

#include <string>
#include <algorithm>
#include "omp.h"
#include "Eva.h"

using namespace Eigen;
using namespace std;

bool add_overlap = false;
bool low_inlieratio = false;
bool no_logs = false;

double OTSU_thresh(/*vector<Vote> Vote_score*/Eigen::VectorXd values)
{
	int i;
	int Quant_num = 100;
	double score_sum = 0.0;
	double fore_score_sum = 0.0;
	vector<int> score_Hist(Quant_num, 0);
	vector<double> score_sum_Hist(Quant_num, 0.0);
	double max_score_value, min_score_value;
	vector<double> all_scores;
	for (i = 0; i < values.size(); i++)
	{
		score_sum += values[i];
		all_scores.push_back(values[i]);
	}
	sort(all_scores.begin(), all_scores.end());
	max_score_value = all_scores[all_scores.size() - 1];
	min_score_value = all_scores[0];
	double Quant_step = (max_score_value - min_score_value) / Quant_num;
    if (fabs(Quant_step) < 1e-12) {
        return min_score_value;
    }
	for (i = 0; i < values.size(); i++)
	{
		int ID = values[i] / Quant_step;
		if (ID >= Quant_num) ID = Quant_num - 1;
		score_Hist[ID]++;
		score_sum_Hist[ID] += values[i];
	}
	double fmax = -1000;
	int n1 = 0, n2;
	double m1, m2, sb;
	double thresh = (max_score_value - min_score_value) / 2;//default value
	for (i = 0; i < Quant_num; i++)
	{
		double Thresh_temp = i * (max_score_value - min_score_value) / double(Quant_num);
		n1 += score_Hist[i];
		if (n1 == 0) continue;
		n2 = values.size() - n1;
		if (n2 == 0) break;
		fore_score_sum += score_sum_Hist[i];
		m1 = fore_score_sum / n1;
		m2 = (score_sum - fore_score_sum) / n2;
		sb = (double)n1 * (double)n2 * pow(m1 - m2, 2);
		if (sb > fmax)
		{
			fmax = sb;
			thresh = Thresh_temp;
		}
	}
	return thresh;
}

//
double Distance(pcl::PointXYZ& A, pcl::PointXYZ& B) {
	double distance = 0;
	double d_x = (double)A.x - (double)B.x;
	double d_y = (double)A.y - (double)B.y;
	double d_z = (double)A.z - (double)B.z;
	distance = sqrt(d_x * d_x + d_y * d_y + d_z * d_z);
	return distance;
}
bool compare_vote_score(const Vote& v1, const Vote& v2) {
	return v1.score > v2.score;
}

bool compare_vote_degree(const Vote_exp& v1, const Vote_exp& v2) {
	return v1.degree > v2.degree;
}

Eigen::MatrixXf Graph_construction(vector<Corre_3DMatch>& correspondence, float resolution, bool sc2, float cmp_thresh) {
    int size = correspondence.size();
    Eigen::MatrixXf cmp_score; 
    cmp_score.resize(size, size);
    cmp_score.setZero();
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < size; i++)
    {
        Corre_3DMatch c1 = correspondence[i];
        for (int j = i + 1; j < size; j++)
        {
            Corre_3DMatch c2 = correspondence[j];
            float score, src_dis, des_dis, dis, alpha_dis;
            src_dis = Distance(c1.src, c2.src);
            des_dis = Distance(c1.des, c2.des);
            dis = abs(src_dis - des_dis);
            alpha_dis = 10 * resolution;
            score = exp(-dis * dis / (2 * alpha_dis * alpha_dis));
            score = (score < cmp_thresh) ? 0 : score;
            cmp_score(i, j) = score;
            cmp_score(j, i) = score;
        }
    }
    if (sc2)
    {
        Eigen::MatrixXf tmp = cmp_score * cmp_score;
        cmp_score = cmp_score.cwiseProduct(tmp);
    }
    return cmp_score;
}

void find_largest_clique_of_node(Eigen::MatrixXf& Graph, igraph_vector_int_list_t* cliques, vector<Corre_3DMatch>& correspondence, node_cliques* result, vector<int>& remain, int num_node, int est_num, string descriptor) {
    int* vis = new int[igraph_vector_int_list_size(cliques)];
    memset(vis, 0, sizeof(int) * igraph_vector_int_list_size(cliques));
#pragma omp parallel for
    for (int i = 0; i < num_node; i++)
    {
        result[i].clique_index = -1;
        result[i].clique_size = 0;
        result[i].clique_weight = 0;
        result[i].clique_num = 0;
    }

    for (int i = 0; i < remain.size(); i++)
    {
        igraph_vector_int_t* v = igraph_vector_int_list_get_ptr(cliques, remain[i]);
        float weight = 0;
        int length = igraph_vector_int_size(v);
        for (int j = 0; j < length; j++)
        {
            int a = (int)VECTOR(*v)[j];
            for (int k = j + 1; k < length; k++)
            {
                int b = (int)VECTOR(*v)[k];
                weight += Graph(a, b);
            }
        }
        for (int j = 0; j < length; j++)
        {
            int k = (int)VECTOR(*v)[j];
            if (result[k].clique_weight < weight)
            {
                result[k].clique_index = remain[i];
                vis[remain[i]]++;
                result[k].clique_size = length;
                result[k].clique_weight = weight;
            }
        }
    }

#pragma omp parallel for
    for (int i = 0; i < 1; i++)
    {
        if (vis[remain[i]] == 0) {
            igraph_vector_int_t* v = igraph_vector_int_list_get_ptr(cliques, remain[i]);
            igraph_vector_int_destroy(v);
        }
    }

    vector<int>after_delete;
    for (int i = 0; i < num_node; i++)
    {
        if (result[i].clique_index < 0)
        {
            continue;
        }
        if (vis[result[i].clique_index] > 0)
        {
            vis[result[i].clique_index] = 0;
            after_delete.push_back(result[i].clique_index);
        }
        else if (vis[result[i].clique_index] == 0) {
            result[i].clique_index = -1;
        }
    }
    remain.clear();
    remain = after_delete;
    if (remain.size() > est_num)
    {
        vector<int>after_decline;
        vector<Vote>clique_score;
        for (int i = 0; i < num_node; i++)
        {
            if (result[i].clique_index < 0)
            {
                continue;
            }
            Vote t;
            t.index = result[i].clique_index;
            t.score = result[i].clique_weight;
            clique_score.push_back(t);
        }
        sort(clique_score.begin(), clique_score.end(), compare_vote_score);
        for (int i = 0; i < est_num; i++)
        {
            after_decline.push_back(clique_score[i].index);
        }
        remain.clear();
        remain = after_decline;
        clique_score.clear();
    }
    delete[] vis;
    return;
}

std::pair<std::vector<int>, float>
intraRCCR(vector<Corre_3DMatch>& correspondence, float resolution, float cmp_thresh) {
    bool sc2 = false;  
    int total_num = correspondence.size();
    int max_est_num = INT_MAX;
    string descriptor = "NULL";

    if (total_num < 3) {
        correspondence.clear();
        correspondence.shrink_to_fit();
        return {std::vector<int>(), 0.0f};
    }

    omp_set_num_threads(12);
    Eigen::MatrixXf Graph = Graph_construction(correspondence, resolution, sc2, cmp_thresh);
    if (!Graph.allFinite() || Graph.norm() == 0) {
        correspondence.clear();
        correspondence.shrink_to_fit();
        return {std::vector<int>(), 0.0f};
    }

    if (total_num==3) {
        float ex_max_clique_weight = -std::numeric_limits<float>::infinity();
        vector<int> ex_top_corr_idx;
        if (Graph(0, 1)>0 && Graph(0,2)>0 && Graph(1,2)>0) {
            ex_top_corr_idx.push_back(0);
            ex_top_corr_idx.push_back(1);
            ex_top_corr_idx.push_back(2);
            ex_max_clique_weight = Graph(0, 1)+Graph(0,2)+Graph(1,2);
        }
        else if (Graph(0, 1)==0 && Graph(0,2)==0 && Graph(1,2)==0) {
            ex_max_clique_weight = 0;
        }
        else {
            if (Graph(0,1)>ex_max_clique_weight) {
                ex_top_corr_idx.push_back(0);
                ex_top_corr_idx.push_back(1);
                ex_max_clique_weight = Graph(0, 1);
            }
            if (Graph(1,2)>ex_max_clique_weight) {
                ex_top_corr_idx.clear();
                ex_top_corr_idx.push_back(2);
                ex_top_corr_idx.push_back(1);
                ex_max_clique_weight = Graph(1, 2);
            }
            if (Graph(0,2)>ex_max_clique_weight) {
                ex_top_corr_idx.clear();
                ex_top_corr_idx.push_back(2);
                ex_top_corr_idx.push_back(0);
                ex_max_clique_weight = Graph(0, 2);
            }
        }
        correspondence.clear();
        correspondence.shrink_to_fit();
        return {ex_top_corr_idx, ex_max_clique_weight};
    }


    vector<int> top_corr_idx;
    float max_clique_weight = 0.0f; 

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < total_num; i++) {
        vector<int> current_clique;
        current_clique.push_back(i);

        vector<pair<float, int>> neighbors;
        for (int j = 0; j < total_num; j++) {
            if (i != j && Graph(i, j) > 0) {
                neighbors.push_back({Graph(i, j), j});
            }
        }

        sort(neighbors.begin(), neighbors.end(), [](const pair<float, int>& a, const pair<float, int>& b) {
            return a.first > b.first;
        });

        for (const auto& neighbor : neighbors) {
            int candidate = neighbor.second;
            bool can_add = true;

            for (int node_in_clique : current_clique) {
                if (Graph(candidate, node_in_clique) == 0) {
                    can_add = false;
                    break;
                }
            }

            if (can_add) {
                current_clique.push_back(candidate);
            }
        }

        int size = current_clique.size();
        
        if (size >= 3) {
            float current_weight = 0;
            for (int x = 0; x < size; x++) {
                for (int y = x + 1; y < size; y++) {
                    current_weight += Graph(current_clique[x], current_clique[y]);
                }
            }

            #pragma omp critical
            {
                if (current_weight > max_clique_weight) {
                    max_clique_weight = current_weight;
                    top_corr_idx = current_clique;
                }
            }
        }
    }

    if (!top_corr_idx.empty()) {
        sort(top_corr_idx.begin(), top_corr_idx.end());
        top_corr_idx.erase(unique(top_corr_idx.begin(), top_corr_idx.end()), top_corr_idx.end());
    }

    correspondence.clear(); correspondence.shrink_to_fit();
   

    return {top_corr_idx, max_clique_weight};
}

int main() {
    Eigen::setNbThreads(6);
}
