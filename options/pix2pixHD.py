class Options:
    def __init__(self):
        # Base options
        self.name = "nir_to_rgb"
        self.gpu_ids = "0"
        self.checkpoints_dir = "models/pix2pixHD/checkpoints"
        self.model = "pix2pixHD"
        self.norm = "instance"
        self.use_dropout = False
        self.data_type = 32
        self.verbose = False
        self.fp16 = False
        self.local_rank = 0

        self.batchSize = 1  # FIXED
        self.loadSize = 256
        self.fineSize = 512
        self.label_nc = 0
        self.input_nc = 3
        self.output_nc = 3

        self.dataroot = "datasets"  # "./../../datasets/nir_to_rgb"
        self.resize_or_crop = "scale_width"
        self.serial_batches = True  # FIXED
        self.no_flip = True  # FIXED
        self.nThreads = 1  # FIXED
        self.max_dataset_size = float("inf")

        self.display_winsize = 512
        self.tf_log = False

        self.netG = "global"
        self.ngf = 64
        self.n_downsample_global = 4
        self.n_blocks_global = 9
        self.n_blocks_local = 3
        self.n_local_enhancers = 1
        self.niter_fix_global = 0

        self.no_instance = True
        self.instance_feat = False
        self.label_feat = False
        self.feat_num = 3
        self.load_features = False
        self.n_downsample_E = 4
        self.nef = 16
        self.n_clusters = 10
        self.isTrain = False

        # Test options
        self.ntest = float("inf")
        self.results_dir = "output/pix2pixHD"
        self.aspect_ratio = 1.0
        self.phase = "test"
        self.which_epoch = "latest"
        self.how_many = float("inf")
        self.cluster_path = "features_clustered_010.npy"
        self.use_encoded_image = False
        self.export_onnx = None
        self.engine = None
        self.onnx = None

        self.parser()

    def parser(self):
        self.gpu_ids = list(map(lambda x: int(x), self.gpu_ids.split(",")))
