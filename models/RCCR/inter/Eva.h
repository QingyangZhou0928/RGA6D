// ------------------------------------------------------------------------------------
// Modified from MAC (https://github.com/zhangxy0517/3D-Registration-with-Maximal-Cliques)
// Originally authored by Xiyu Zhang (Copyright (c) 2023 zhangxy0517)
// Licensed under the MIT License.
// Modifications copyright (c) 2026 Qingyang Zhou
// ------------------------------------------------------------------------------------
#ifndef _EVA_H_ 
#define _EVA_H_
#define Pi 3.1415926
#define constE 2.718282
#define NULL_POINTID -1
#define NULL_Saliency -1000
#define Corres_view_gap -200
#define Align_precision_threshold 0.1
#define tR 116//30
#define tG 205//144
#define tB 211//255
#define sR 253//209//220
#define sG 224//26//20
#define sB 2//32//60
#define L2_thresh 0.5
#define Ratio_thresh 0.2
#define GC_dist_thresh 3
#define Hough_bin_num 15
#define SI_GC_thresh 0.8
#define RANSAC_Iter_Num 5000
#define GTM_Iter_Num 100
#define CV_voting_size 20
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
extern bool add_overlap;
extern bool low_inlieratio;
extern bool no_logs;
//
#include <pcl/surface/gp3.h>
#include <pcl/surface/mls.h>
#include <unordered_set>
#include <Eigen/Eigen>
#include <igraph/igraph.h>
#include <sys/stat.h>
#include <unistd.h>
using namespace std;
//
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;
typedef pcl::PointXYZ PointInT;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatD;
typedef struct {
	float x;
	float y;
	float z;
}Vertex;
typedef struct {
	float x;
	float y;
	float z;
	float dist;
	float angle_to_axis;
}Vertex_d_ang;
typedef struct {
	int pointID;
	Vertex x_axis;
	Vertex y_axis;
	Vertex z_axis;
}LRF;
typedef struct {
	int source_idx;
	int target_idx;
	LRF source_LRF;
	LRF target_LRF;
	double score;
}Corre;
typedef struct {
	int src_index;
	int des_index;
	pcl::PointXYZ src;
	pcl::PointXYZ des;
	Eigen::Vector3f src_norm;
	Eigen::Vector3f des_norm;
	Eigen::Matrix3f covariance_src, covariance_des;					
	Eigen::Vector4f centeroid_src, centeroid_des;				
	double score;
	int inlier_weight;
}Corre_3DMatch;
typedef struct {
	int PointID;
	float eig1_2;
	float eig2_3;
	float saliency;
	bool TorF;
}ISS_Key_Type;
typedef struct {
	float M[4][4];
}TransMat;
typedef struct
{
	int index;
	double score;
}Vote;
typedef struct
{
	int index;
	int degree;
	double score;
	vector<int> corre_index;
	int true_num;
}Vote_exp;
typedef struct
{
	vector<int> v;
	int pt1;
	int pt2;
}Intersection_set;
typedef struct
{
	int clique_index;
	int clique_size;
	float clique_weight;
	int clique_num;
}node_cliques;

double OTSU_thresh(/*vector<Vote> Vote_score*/Eigen::VectorXd values);
double Distance(pcl::PointXYZ& A, pcl::PointXYZ& B);
bool compare_vote_score(const Vote& v1, const Vote& v2);
bool compare_vote_degree(const Vote_exp& v1, const Vote_exp& v2);
Eigen::MatrixXf Graph_construction(vector<Corre_3DMatch>& correspondence, float resolution, bool sc2, float cmp_thresh);
void find_largest_clique_of_node(Eigen::MatrixXf& Graph, igraph_vector_int_list_t* cliques, vector<Corre_3DMatch>& correspondence, node_cliques* result, vector<int>& remain, int num_node, int est_num, string descriptor);
std::tuple<std::vector<int>, std::vector<float>> interRCCR(vector<Corre_3DMatch>& correspondence, vector<Corre_3DMatch>& all_correspondences, torch::Tensor& region_to_corr_indices, float resolution, float cmp_thresh);
#endif
