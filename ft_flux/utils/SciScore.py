import torch
from transformers import AutoProcessor, AutoModel
from groundingdino.util.inference import load_model, predict


def find_enclosing_box_cxcywh(box_tensor):
    if box_tensor.numel() == 0:
        return torch.tensor([0.5, 0.5, 1.0, 1.0], device=box_tensor.device, dtype=box_tensor.dtype)
    
    x_mins = box_tensor[:, 0] - box_tensor[:, 2] / 2
    y_mins = box_tensor[:, 1] - box_tensor[:, 3] / 2
    x_maxs = box_tensor[:, 0] + box_tensor[:, 2] / 2
    y_maxs = box_tensor[:, 1] + box_tensor[:, 3] / 2
    
    enclosing_x_min = x_mins.min()
    enclosing_y_min = y_mins.min()
    enclosing_x_max = x_maxs.max()
    enclosing_y_max = y_maxs.max()
    
    enclosing_cx = (enclosing_x_min + enclosing_x_max) / 2
    enclosing_cy = (enclosing_y_min + enclosing_y_max) / 2
    enclosing_w = enclosing_x_max - enclosing_x_min
    enclosing_h = enclosing_y_max - enclosing_y_min
    
    enclosing_w = torch.where(enclosing_w <= 0.9, enclosing_w + 0.1, torch.tensor(1.0, device=enclosing_w.device))
    enclosing_h = torch.where(enclosing_h <= 0.9, enclosing_h + 0.1, torch.tensor(1.0, device=enclosing_h.device))
    
    enclosing_box = torch.stack([enclosing_cx, enclosing_cy, enclosing_w, enclosing_h])
    return enclosing_box


class SciScore(torch.nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.processor = AutoProcessor.from_pretrained("Jialuo21/SciScore")
        self.model = AutoModel.from_pretrained('Jialuo21/SciScore').eval().to(self.device, dtype=self.dtype)
        self.dino_model = load_model("GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", 
                                     "GroundingDINO/weights/groundingdino_swint_ogc.pth", 
                                     device=self.device)
        self.BOX_THRESHOLD = 0.35
        self.TEXT_THRESHOLD = 0.25

    @torch.no_grad()
    def __call__(self, images, prompts, objects):
        
        box_list = []
        for i in range(len(prompts)):
            boxes, logits, phrases = predict(
                model=self.dino_model,
                image=images[i].to(torch.float32),
                caption=objects[i],
                box_threshold=self.BOX_THRESHOLD,
                text_threshold=self.TEXT_THRESHOLD
            )
            box = find_enclosing_box_cxcywh(boxes).cpu()  
            box_list.append(box)
        
        box_tensor = torch.stack(box_list)  # [batchsize, 4]


        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        image_embeds = self.model.get_image_features(**image_inputs)
        image_embeds = image_embeds / torch.norm(image_embeds, dim=-1, keepdim=True)
        text_embeds = self.model.get_text_features(**text_inputs)
        text_embeds = text_embeds / torch.norm(text_embeds, dim=-1, keepdim=True)
        logits_per_image = image_embeds @ text_embeds.T
        scores = torch.diagonal(logits_per_image) * self.model.logit_scale.exp()
        # print(scores)
        return scores, box_tensor