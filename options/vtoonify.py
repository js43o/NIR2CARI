class Options:
    def __init__(self):
        self.content = None
        self.input_path = "temp_output"
        self.ckpt = "models/VToonify/checkpoint/vtoonify_t.pt"
        self.output_path = "output"
        self.scale_image = True
        self.style_encoder_path = "models/VToonify/checkpoint/encoder.pt"
        self.faceparsing_path = "models/VToonify/checkpoint/faceparsing.pth"
        self.cpu = False
        self.padding = [200, 200, 200, 200]
        self.batch_size = 1
        self.parsing_map_path = None
