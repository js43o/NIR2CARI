class Options:
    def __init__(self):
        self.content = None  # "temp_output/Aaron_Peirsol_0001_rgb.jpg"
        self.input_path = "output/pix2pixHD"
        self.ckpt = "models/VToonify/checkpoint/vtoonify_t.pt"
        self.output_path = "output/vtoonify"
        self.scale_image = True
        self.style_encoder_path = "models/VToonify/checkpoint/encoder.pt"
        self.faceparsing_path = "models/VToonify/checkpoint/faceparsing.pth"
        self.cpu = False
        self.padding = [200, 200, 200, 200]
        self.batch_size = 1
        self.parsing_map_path = None
