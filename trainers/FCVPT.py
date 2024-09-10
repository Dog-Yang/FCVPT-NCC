from torch.nn import Conv2d, Dropout
import math
import os.path as osp
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_checkpoint
from dassl.data.data_manager import DataManager
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.clip import tokenize
_tokenizer = _Tokenizer()
from utils.model_utils import *
from utils.utils import *
_tokenizer = _Tokenizer()
from functools import reduce
from operator import mul
from utils.data_utils import ds_specific_templates

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root='all_weights')

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class FCVPT_Model(nn.Module):

    def __init__(self, model, classes, templates, ds_templates=None, device='cuda', log=None, dataset_name=None, txt_cls=None, cfg=None):
        super(FCVPT_Model, self).__init__()
        self.cfg = cfg
        self.log = log
        self.device = device
        self.model = model.to(device)
        self.backbone = cfg.MODEL.BACKBONE.NAME.replace('/', '')
        
        self.classes = classes
        self.templates = templates
        self.dataset_name = dataset_name
        self.dataset_templates = ds_templates
        
        patch_size = (16, 16)
        self.backbone_out_size = 512
        self.hidden_size = 768   # todo for ViT-B use 768 - for ViT-L use 1024 
        self.hidden_size_text = 512
        self.num_tokens = 50     # todo all experiments are run with num_tokens = 50
        self.num_tokens_text = 77
        
        # prompt parameters
        prompt_dim = self.hidden_size
        self.prompt_proj = nn.Identity()
        self.prompt_dropout = Dropout(0.0)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
        
        # parametric classifier
        self.pcl_student = nn.Sequential(nn.Linear(int(self.backbone_out_size), len(classes), bias=False)).to(device)
        self.pcl_teacher = None
        
        # visual prompt embeddings
        self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.num_tokens, self.hidden_size), requires_grad=True)
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        
        # CLIP text classifier and  LLM orpus classifier
        self.txt_cls = txt_cls
        self.txt_features_for_text_cls, self.labels_for_text_cls = self.txt_features_for_text_cls()
        self.text_features = self.txt_features()

    def train_txt_clas(self, criteria):
        noise_std = 0.1
        noise = torch.randn(self.txt_features_for_text_cls.shape) * noise_std
        txt_feas = self.txt_features_for_text_cls
        txt_label = self.labels_for_text_cls
        feas = (self.pcl_student(txt_feas.to(torch.float32) + noise.cuda()))
        loss = criteria(feas, txt_label)
        return loss

    def txt_features_for_text_cls(self):
        if self.txt_cls == 'CLIP':
            gpt3_prompts = self.templates
            desc, labels_for_descriptions = gen_labels_with_templates(self.classes, descriptions=gpt3_prompts)

        elif self.txt_cls == 'LLM':
            # generic prompts + templates
            if self.dataset_name not in FCVPT_datasets:
                raise ValueError('Invalid dataset name')

            path_to_file = f'./descriptions/generic/{self.dataset_name}.json'
            with open(path_to_file) as f:
                gpt3_prompts = json.load(f)

            desc, labels_for_descriptions = gen_labels_with_descrptions(self.classes, descriptions=gpt3_prompts)
            templates, labels_for_templates = gen_labels_with_templates(self.classes, descriptions=self.dataset_templates)

            desc += templates
            labels_for_descriptions += labels_for_templates

        elif self.txt_cls == 'zero_shot':
            pass

        else:
            raise ValueError('Invalid txt_cls argument')


        if self.txt_cls in ['CLIP', 'LLM']:

            Path(f'embeddings').mkdir(parents=True, exist_ok=True)

            if os.path.isfile(f'embeddings/{self.backbone}_{self.txt_cls}_{self.dataset_name}.pt'):
                zeroshot_weights = torch.load(f'embeddings/{self.backbone}_{self.txt_cls}_{self.dataset_name}.pt')
                print('******** Loaded Already Saved Embeddings *********')
                labels_for_descriptions = torch.tensor(labels_for_descriptions).cuda()

            else:
                print('******** No Embeddings Found --- Saving New Embeddings *********')

                labels_for_descriptions = torch.tensor(labels_for_descriptions).cuda()

                zeroshot_weights = []
                with torch.no_grad():
                    for classname in tqdm(desc):
                        text = tokenize(classname).cuda()  # tokenize # (50, 77) --> 50 templates/texts from GPT
                        class_embeddings = self.model.encode_text(text)  # embed with text encoder # (50, 512) --> embeddings for all 50 texts
                        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)  # L2 norm of the embeddings (dim 2)
                        zeroshot_weights.append(class_embeddings)
                    zeroshot_weights = torch.stack(zeroshot_weights).cuda()  # (512, 10) --> 512 embeddings for 10 classes'
                    torch.save(zeroshot_weights, f'embeddings/{self.backbone}_{self.txt_cls}_{self.dataset_name}.pt')

            return zeroshot_weights.squeeze(), labels_for_descriptions

        else:
            return None, None

    def txt_features(self):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(self.classes):
                texts = [template.format(classname) for template in self.templates]  # format with class
                texts = tokenize(texts).cuda()  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    def image_features(self, images):
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features

    def eval_clip(self, x):
        with torch.no_grad():
            img_features_2 = self.incorporate_prompt(x)
            img_features_2 = self.embeddings_after_prompts(img_features_2)
            img_features_pcl = self.pcl_student(img_features_2)
        return img_features_pcl, img_features_2

    def forward(self, x):
        # only used for 0-shot-eval
        with torch.no_grad():
            img_features = self.image_features(x)
            pseudo_label = img_features @ self.text_features
        return pseudo_label, img_features

    def forward_student(self, x2):
        '''
        :param x1: the clean image (without transforms, for pseudo labels, for teacher)
        :param x2: the transformed image (for student)
        :return: features pcl (cls head), pseudo-labels
        '''
        img_features_2 = self.incorporate_prompt(x2)
        img_features_2 = self.embeddings_after_prompts(img_features_2)
        img_features_pcl = self.pcl_student(img_features_2)
        return img_features_pcl

    def txt_cls_init(self):
        import copy
        self.pcl_teacher = copy.deepcopy(self.pcl_student)
        
    def forward_teacher(self, x1):
        '''
        :param x1: the clean image (without transforms, for pseudo labels, for teacher)
        :param x2: the transformed image (for student)
        :return: features pcl (cls head), pseudo-labels
        '''
        with torch.no_grad():
            img_features_1 = self.image_features(x1)
            pseudo_label = self.pcl_teacher(img_features_1.float()).detach()
        return pseudo_label, img_features_1

    def incorporate_prompt(self, x):
        B = x.shape[0]
        x = self.patch_embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        x = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
            x[:, 1:, :]
        ), dim=1)
        return x
    
    def patch_embeddings(self, x: torch.tensor):
        return self.model.visual.embeddings_patch(x)

    def embeddings_after_prompts(self, x: torch.tensor):
        return self.model.visual.forward_after_patch_embeddings(x)

    def positional_embeddings_for_text(self, x: torch.tensor):
        return self.model.positional_embeddings(x)

    def embeddings_after_prompts_for_text(self, x: torch.tensor):
        return self.model.embeddings_after_prompting(x)


@TRAINER_REGISTRY.register()
class FCVPT(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()
        print("Building ZERO-SHOT-MODEL CLIP")
        self.model = FCVPT_Model(model=clip_model, classes=classnames, dataset_name= cfg.DATASET.NAME, txt_cls = cfg.txt_cls, cfg=cfg,
                                 templates=['a photo of a {}'], ds_templates = ds_specific_templates[cfg.DATASET.NAME])
        self.register_model("adapt", self.model)
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg, custom_tfm_test=te_transform, custom_tfm_train=tr_transforms)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def parse_batch_train(self, batch):

        if isinstance(batch, list):
            input = batch["img"]
            input = torch.stack(input)  # two views from dataloader
            input = input.to(self.device)
        else:
            input = batch['img']
            input = input.to(self.device)

        label = batch["label"]
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
