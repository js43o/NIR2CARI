class Options:
    def __init__(self):
        self.gpu_ids = "0"
        self.model = "pix2pixHD"
        self.dataroot = "datasets"  # "./../../datasets/nir_to_rgb"
        self.parser()

    def parser(self):
        self.gpu_ids = list(map(lambda x: int(x), self.gpu_ids.split(",")))
