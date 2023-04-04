import sys
from pathlib import Path
from pprint import pformat

from hloc import extract_features, match_features, pairs_from_covisibility, pairs_from_retrieval
from hloc import colmap_from_nvm, triangulation, localize_sfm, visualization

class VLPBenchmark():
    def __init__(self) -> None:
        self.dataset = Path("/mnt/nas/share-all/caizebin/03.dataset/opensource/datasets/aachen")
        self.outputs = Path("/mnt/nas/share-all/caizebin/03.dataset/opensource/output/aachen")  # where everything will be saved
        self.initialize()


    def initialize(self):
        self.images = self.dataset / 'images/images_upright/'
        self.sfm_pairs = self.outputs / 'pairs-db-covis20.txt'  # top 20 most covisible in SIFT model
        self.loc_pairs = self.outputs / 'pairs-query-netvlad20.txt'  # top 20 retrieved by NetVLAD
        self.reference_sfm = self.outputs / 'sfm_superpoint+superglue'  # the SfM model we will build
        self.results =  self.outputs / 'Aachen_hloc_superpoint+superglue_netvlad20.txt'  # the result file
        
        self.feature_conf = extract_features.confs['superpoint_aachen']
        self.matcher_conf = match_features.confs['superglue']
        self.feature_path =  Path(self.outputs, self.feature_conf['output']+'.h5')
        self.match_path =   Path(self.outputs, f'{self.feature_conf["output"]}_{self.matcher_conf["output"]}_{self.sfm_pairs.stem}.h5')

        # list the standard configurations available
        print(f'Configs for feature extractors:\n{pformat(extract_features.confs)}')
        print(f'Configs for feature matchers:\n{pformat(match_features.confs)}')
    
    def feature_extraction(self):
        features = extract_features.main(self.feature_conf, self.images, self.outputs)
        
    
    def generate_pairs(self):
        colmap_from_nvm.main(
            self.dataset / '3D-models/aachen_cvpr2018_db.nvm',
            self.dataset / '3D-models/database_intrinsics.txt',
            self.dataset / 'aachen.db',
            self.outputs / 'sfm_sift')

        pairs_from_covisibility.main(
            self.outputs / 'sfm_sift', self.sfm_pairs, num_matched=20)
    
    def match_feature(self):
        matcher_conf = match_features.confs['superglue']
        feature_conf = extract_features.confs['superpoint_aachen']
        sfm_matches = match_features.main(matcher_conf,  self.sfm_pairs, self.feature_conf['output'], self.outputs)
        
    
    def triangulation(self):
        reconstruction = triangulation.main(
            self.reference_sfm,
            self.outputs / 'sfm_sift',
            self.images,
            self.sfm_pairs,
            self.feature_path,
            self.match_path)
        
    def localization_image_retrieval(self):
        retrieval_conf = extract_features.confs['netvlad']
        global_descriptors = extract_features.main(retrieval_conf, self.images, self.outputs)
        pairs_from_retrieval.main(global_descriptors, self.loc_pairs, num_matched=20, db_prefix="db", query_prefix="query")
        
    def localization_matching(self):
        loc_matches = match_features.main(self.matcher_conf, self.loc_pairs, self.feature_conf['output'], self.outputs)
        print(loc_matches)
    
    def localization(self):
        reconstruction = "/mnt/nas/share-all/caizebin/03.dataset/opensource/output/aachen/sfm_superpoint+superglue"
        loc_matches = Path("/mnt/nas/share-all/caizebin/03.dataset/opensource/output/aachen/feats-superpoint-n4096-r1024_matches-superglue_pairs-query-netvlad20.h5")
        localize_sfm.main(
            reconstruction,
            self.dataset / 'queries/*_time_queries_with_intrinsics.txt',
            self.loc_pairs,
            self.feature_path,
            loc_matches,
            self.results,
            covisibility_clustering=False)  # not required with SuperPoint+SuperGlue
    
    def visualize_localization(self):
        reconstruction = "/mnt/nas/share-all/caizebin/03.dataset/opensource/output/aachen/sfm_superpoint+superglue"
        visualization.visualize_loc(
            self.results, self.images, reconstruction, n=1, top_k_db=1, prefix='query/night', seed=2)
    
    

def test_feature_extraction():
    benchmark = VLPBenchmark()
    benchmark.feature_extraction()
    

def test_generate_pairs():
    benchmark = VLPBenchmark()
    benchmark.generate_pairs()
    

def test_match_feature():
    benchmark = VLPBenchmark()
    benchmark.match_feature()
    

def test_triangulation():
    benchmark = VLPBenchmark()
    benchmark.triangulation()
    

def test_localization_image_retrieval():
    benchmark = VLPBenchmark()
    benchmark.localization_image_retrieval()


def test_localization_matching():
    benchmark = VLPBenchmark()
    benchmark.localization_matching()


def test_localization():
    benchmark = VLPBenchmark()
    benchmark.localization()


def test_visualize_localization():
    benchmark = VLPBenchmark()
    benchmark.visualize_localization()

    