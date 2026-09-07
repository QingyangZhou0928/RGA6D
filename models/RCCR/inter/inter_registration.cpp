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
#include <torch/extension.h>
#include <stdio.h>
#include <time.h>
#include <random>

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
    for (int i = 0; i < size; i++)
    {
        Corre_3DMatch c1 = correspondence[i];
        for (int j = i + 1; j < size; j++)
        {
            Corre_3DMatch c2 = correspondence[j];
            float src_dis, des_dis, dis, alpha_dis, score;
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

std::tuple<std::vector<int>, std::vector<float>>
interRCCR(vector<Corre_3DMatch>& correspondence, 
    vector<Corre_3DMatch>& all_correspondences, 
    torch::Tensor& region_to_corr_indices, 
    float resolution, float cmp_thresh) {
    bool sc2 = false;  
    int region_num = correspondence.size();
    int max_est_num = INT_MAX;
    string descriptor = "NULL";

    if (region_num < 2) {
        correspondence.clear();
        correspondence.shrink_to_fit();
        return {std::vector<int>(), std::vector<float>()};
    }

    Eigen::MatrixXf Graph = Graph_construction(correspondence, resolution, sc2, cmp_thresh);
    if (!Graph.allFinite() || Graph.norm() == 0) {
        correspondence.clear();
        correspondence.shrink_to_fit();
        return {std::vector<int>(), std::vector<float>()};
    }

    int num_trials = 5;      
    int num_samples = 3;     
    int corre_num = all_correspondences.size();
    auto region_to_corr_indices_accessor = region_to_corr_indices.accessor<int, 2>();
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < region_num; i++) {
        for (int j = i + 1; j < region_num; j++) {
            if (Graph(i, j) == 0) continue; 

            vector<int> idx_i, idx_j;
            for(int k=0; k<corre_num; k++) {
                if(region_to_corr_indices_accessor[i][k] != -1) idx_i.push_back(region_to_corr_indices_accessor[i][k]);
                if(region_to_corr_indices_accessor[j][k] != -1) idx_j.push_back(region_to_corr_indices_accessor[j][k]);
            }
            if (idx_i.size() < num_samples || idx_j.size() < num_samples) {
                Graph(i, j) = Graph(j, i) = 0;
                continue;
            }

            bool edge_valid = false;
            std::random_device rd;
            std::mt19937 g(rd());
            for (int t = 0; t < num_trials; t++) {
                std::shuffle(idx_i.begin(), idx_i.end(), g);
                std::shuffle(idx_j.begin(), idx_j.end(), g);
                vector<Corre_3DMatch> sampled_test;
                for(int s=0; s<num_samples; s++) {
                    sampled_test.push_back(all_correspondences[idx_i[s]]);
                    sampled_test.push_back(all_correspondences[idx_j[s]]);
                }
                bool trial_pass = true;
                for (int m = 0; m < 6; m++) {
                    for (int n = m + 1; n < 6; n++) {
                        float src_dis = Distance(sampled_test[m].src, sampled_test[n].src);
                        float des_dis = Distance(sampled_test[m].des, sampled_test[n].des);
                        float dis_diff = abs(src_dis - des_dis);
                        float alpha = 10 * resolution;
                        float score = exp(-dis_diff * dis_diff / (2 * alpha * alpha));
                        
                        if (score < cmp_thresh) {
                            trial_pass = false;
                            break;
                        }
                    }
                    if (!trial_pass) break;
                }

                if (trial_pass) {
                    edge_valid = true;
                    break; 
                }
            }

            if (!edge_valid) {
                Graph(i, j) = Graph(j, i) = 0; 
            }
        }
    }

    vector<int>degree(region_num, 0);
    vector<Vote_exp> pts_degree;
    for (int i = 0; i < region_num; i++)
    {
        Vote_exp t;
        t.true_num = 0;
        vector<int> corre_index;
        for (int j = 0; j < region_num; j++)
        {
            if (i != j && Graph(i, j)) {
                degree[i] ++;
                corre_index.push_back(j);
            }
        }
        t.index = i;  
        t.degree = degree[i];  
        t.corre_index = corre_index;  
        pts_degree.push_back(t);
    }

    vector<Vote> cluster_factor;
    double sum_fenzi = 0;
    double sum_fenmu = 0;
    omp_set_num_threads(12); 
    for (int i = 0; i < region_num; i++)
    {
        Vote t;
        double sum_i = 0;
        double wijk = 0;
        int index_size = pts_degree[i].corre_index.size();
#pragma omp parallel
        {
#pragma omp for
            for (int j = 0; j < index_size; j++)
            {
                int a = pts_degree[i].corre_index[j];
                for (int k = j + 1; k < index_size; k++)
                {
                    int b = pts_degree[i].corre_index[k];
                    if (Graph(a, b)) {
#pragma omp critical
                        wijk += pow(Graph(i, a) * Graph(i, b) * Graph(a, b), 1.0 / 3); 
                    }
                }
            }
        }
        if (degree[i] > 1)
        {
            double f1 = wijk;
            double f2 = degree[i] * (degree[i] - 1) * 0.5;
            sum_fenzi += f1;
            sum_fenmu += f2;
            double factor = f1 / f2;  
            t.index = i;
            t.score = factor;
            cluster_factor.push_back(t);
        }
        else { 
            t.index = i;
            t.score = 0;
            cluster_factor.push_back(t);
        }
    }
    
    double average_factor = 0;
    for (size_t i = 0; i < cluster_factor.size(); i++)
    {
        average_factor += cluster_factor[i].score;
    }
    average_factor /= cluster_factor.size();
    if (cluster_factor.empty())
    {
        correspondence.clear();
        correspondence.shrink_to_fit();
        degree.clear();
        degree.shrink_to_fit();
        pts_degree.clear();
        pts_degree.shrink_to_fit();
        cluster_factor.clear();
        cluster_factor.shrink_to_fit();
        return {std::vector<int>(), std::vector<float>()};
    }

    if (sum_fenmu == 0)
    {
        correspondence.clear();
        correspondence.shrink_to_fit();
        degree.clear();
        degree.shrink_to_fit();
        pts_degree.clear();
        pts_degree.shrink_to_fit();
        cluster_factor.clear();
        cluster_factor.shrink_to_fit();
        return {std::vector<int>(), std::vector<float>()};
    }
    double total_factor = sum_fenzi / sum_fenmu;

    vector<Vote_exp> pts_degree_bac;  
    vector<Vote>cluster_factor_bac;  
    pts_degree_bac.assign(pts_degree.begin(), pts_degree.end());
    cluster_factor_bac.assign(cluster_factor.begin(), cluster_factor.end());

    sort(cluster_factor.begin(), cluster_factor.end(), compare_vote_score); 
    sort(pts_degree.begin(), pts_degree.end(), compare_vote_degree); 

    Eigen::VectorXd cluster_coefficients;
    cluster_coefficients.resize(cluster_factor.size());
    for (size_t i = 0; i < cluster_factor.size(); i++)
    {
        cluster_coefficients[i] = cluster_factor[i].score;
    }

    int cnt = 0;
    double OTSU = 0;
    if (cluster_factor[0].score != 0)  
    {
        OTSU = OTSU_thresh(cluster_coefficients);
    }
    double cluster_threshold = min(OTSU, min(average_factor, total_factor));

    double weight_thresh = cluster_threshold;
    if (add_overlap)
    {
        weight_thresh = 0.5;
    }
    else {
        weight_thresh = 0;
    }

    if (!add_overlap)
    {
        for (size_t i = 0; i < region_num; i++)
        {
            correspondence[i].score = cluster_factor_bac[i].score;
        }
    }
    
    igraph_t g;
    igraph_matrix_t g_mat;
    igraph_vector_t weights;
    igraph_vector_init(&weights, Graph.rows() * (Graph.cols() - 1) / 2); 
    igraph_matrix_init(&g_mat, Graph.rows(), Graph.cols());
    if (cluster_threshold > 3 && correspondence.size() > 100) 
    {
        float f = 10;
        while (1)
        {
            if (f * max(OTSU, total_factor) > cluster_factor[49].score)
            {
                f -= 0.05;
            }
            else {
                break;
            }
        }
        for (int i = 0; i < Graph.rows(); i++)
        {
            if (cluster_factor_bac[i].score > f * max(OTSU, total_factor))
            {
                for (int j = i + 1; j < Graph.cols(); j++)
                {
                    if (cluster_factor_bac[j].score > f * max(OTSU, total_factor))
                    {
                        MATRIX(g_mat, i, j) = Graph(i, j);
                        MATRIX(g_mat, j, i) = MATRIX(g_mat, i, j);
                    }
                }
            }
        }
    }
    else {
        for (int i = 0; i < Graph.rows(); i++)
        {
            for (int j = i + 1; j < Graph.cols(); j++)
            {
                if (Graph(i, j))
                {
                    MATRIX(g_mat, i, j) = Graph(i, j);
                    MATRIX(g_mat, j, i) = MATRIX(g_mat, i, j);
                }
            }
        }
    } 

    igraph_set_attribute_table(&igraph_cattribute_table);
    igraph_vector_t weight;
    igraph_vector_init(&weight, 0);
    igraph_weighted_adjacency(&g, &g_mat, IGRAPH_ADJ_UNDIRECTED, &weight, IGRAPH_LOOPS_ONCE);


    igraph_vector_int_list_t cliques;
    igraph_vector_int_list_init(&cliques, 0);

    int min_clique_size = 3;
    int max_clique_size = 0;
    bool recomputecliques = true;
    int clique_num = 0; 
    int iter_num = 1;

    igraph_maximal_cliques(&g, &cliques, min_clique_size,  max_clique_size); 
    clique_num = igraph_vector_int_list_size(&cliques);

    if (clique_num == 0) {
        correspondence.clear();
        correspondence.shrink_to_fit();
        degree.clear();
        degree.shrink_to_fit();
        pts_degree.clear();
        pts_degree.shrink_to_fit();
        pts_degree_bac.clear();
        pts_degree_bac.shrink_to_fit();
        cluster_factor.clear();
        cluster_factor.shrink_to_fit();
        cluster_factor_bac.clear();
        cluster_factor_bac.shrink_to_fit();
        igraph_vector_int_list_destroy(&cliques);
        igraph_vector_destroy(&weights);
        igraph_vector_destroy(&weight);
        return {std::vector<int>(), std::vector<float>()};
    }

    igraph_destroy(&g);
    igraph_matrix_destroy(&g_mat);

    vector<int>remain;
    for (int i = 0; i < clique_num; i++)
    {
        remain.push_back(i);
    }
    node_cliques* N_C = new node_cliques[(int)region_num];
    find_largest_clique_of_node(Graph, &cliques, correspondence, N_C, remain, region_num, max_est_num, descriptor);

    vector<int> node_clique(region_num, -1);
    vector<float> node_weight(region_num, 0);
#pragma omp parallel for
    for (int i = 0; i < remain.size(); i++)
    {
        igraph_vector_int_t* v = igraph_vector_int_list_get_ptr(&cliques, remain[i]);
        int group_size = igraph_vector_int_size(v);
        for (int j = 0; j < group_size; j++)
        {
            int node = VECTOR(*v)[j];
            node_clique[node] = remain[i];
            node_weight[node] = N_C[j].clique_weight;
        }
    }

    correspondence.clear();
    correspondence.shrink_to_fit();
    degree.clear();
    degree.shrink_to_fit();
    pts_degree.clear();
    pts_degree.shrink_to_fit();
    pts_degree_bac.clear();
    pts_degree_bac.shrink_to_fit();
    cluster_factor.clear();
    cluster_factor.shrink_to_fit();
    cluster_factor_bac.clear();
    cluster_factor_bac.shrink_to_fit();
    delete[] N_C;
    remain.clear();
    remain.shrink_to_fit();
    igraph_vector_int_list_destroy(&cliques);
    igraph_vector_destroy(&weights);
    igraph_vector_destroy(&weight);

    return {node_clique, node_weight};
}

