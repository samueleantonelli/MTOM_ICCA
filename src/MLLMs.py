import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "LLaVA")))

from itertools import chain
from PIL import Image
import time, copy, random
import httpx
from abc import ABC, abstractmethod




class ModelWrapper(ABC):
    def __init__(self, model_args):
        self.model_args = model_args
        print("model_ckpt:", model_args.model_ckpt)

    @abstractmethod
    def get_spkr_intro(self, context_imgs):
        pass

    @abstractmethod
    def get_lsnr_intro(self):
        pass

    def _model_specific_prompt_postprocessing(self, prompt):
        return prompt  # can override this as needed

    def get_spkr_prompt(
        self, intro, t, context_imgs, target_fn, interaction_args, records=[]
    ):
        label_space = self.model_args.label_space
        history = [entry["spkr_trial_record"] for entry in records[:t]] if t > 0 else []
        for i in range(4):
            if context_imgs[i]["filename"] == target_fn:
                target_label = label_space[i]
                target_img = context_imgs[i]
                break

        if interaction_args.no_history:
            round_name = "Current Round"
        else:
            round_name = f"Round {t+1}"

        trial_prompt = self._get_spkr_prompt(round_name, target_label)

        history = list(chain.from_iterable(history))

        if intro not in history and interaction_args.has_intro:
            history = intro + history

        prompt = history + trial_prompt

        prompt = self._model_specific_prompt_postprocessing(prompt)
        return prompt, trial_prompt, target_img, target_label, context_imgs

    def get_lsnr_prompt(
        self,
        intro,
        t,
        context_imgs,
        target_fn,
        msg,
        records=[],
        random_seed=None,
        no_history=False,
        do_shuffle=False,
        omit_img=False,
        misleading=False,
        has_intro=True,
    ):
        if no_history:
            history = []

        else:
            history = (
                [entry["lsnr_trial_record"] for entry in records[:t]] if t > 0 else []
            )

        trial_imgs = context_imgs.copy()
        if do_shuffle:
            random.seed(random_seed)
            random.shuffle(trial_imgs)

        label_space = self.model_args.label_space
        for i in range(4):
            if trial_imgs[i]["filename"] == target_fn:
                target_img = trial_imgs[i]
                target_label = label_space[i]
                break

        trial_imgs_after_1st_shuffle = trial_imgs.copy()  # in the misleading manipulation, the gold label will be based on this order but the images are shuffled again

        if misleading:
            random.seed(t + 1)
            random.shuffle(trial_imgs)

        if no_history:
            round_name = "Current Round"
        else:
            round_name = f"Round {t+1}"

        trial_prompt = self._get_lsnr_prompt(round_name, trial_imgs, msg, omit_img)

        history = list(chain.from_iterable(history))

        if intro not in history and has_intro:
            history = intro + history

        prompt = history + trial_prompt

        prompt = self._model_specific_prompt_postprocessing(prompt)
        return (
            prompt,
            trial_prompt,
            target_img,
            target_label,
            trial_imgs_after_1st_shuffle,
            trial_imgs,
        )

    @abstractmethod
    def _get_spkr_prompt(self, round_name, target_label):
        pass

    @abstractmethod
    def _get_lsnr_prompt(self, round_name, trial_imgs, msg, omit_img):
        pass

    @abstractmethod
    def query(self, query):
        pass

    @abstractmethod
    def update_with_spkr_pred(self, spkr_trial_prompt, spkr_pred):
        pass

    @abstractmethod
    def update_with_lsnr_pred(self, lsnr_trial_prompt, lsnr_pred):
        pass

    # how the feedback is presented is model-dependent and can be thought of as a hyperparameter. can try different phrasing/formats.
    def get_spkr_feedback(self, pred_fn, spkr_tgt_img, spkr_trial_imgs):
        return self._get_spkr_feedback(pred_fn, spkr_tgt_img, spkr_trial_imgs)

    def get_lsnr_feedback(self, pred, target_img, context_imgs, spkr_msg):
        target_label = self.model_args.label_space[
            [img["filename"] for img in context_imgs].index(target_img["filename"])
        ]
        return self._get_lsnr_feedback(pred, target_label, spkr_msg)

    @abstractmethod
    def _get_lsnr_feedback(self, pred, target_label, spkr_msg):
        pass


class LlavaModel(ModelWrapper):
    def __init__(self, model_args, loaded_model=None):
        self.model_args = model_args
        from LLaVA.llava.model.builder import load_pretrained_model 
        from LLaVA.llava.mm_utils import get_model_name_from_path
        from LLaVA.llava.eval.run_llava import eval_model

        self.eval_model = eval_model

        model_path = model_args.model_ckpt

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                model_path=model_path, model_base=None, model_name=self.model_name
            )
        )


    def query_mtom(self, prompt, trial_imgs):
        
        #    Query the MToM model to refine messages or predictions.

        mtom_query = "".join(prompt)  #process MToM prompt
        merged_context = merge_images([img[self.model_args.img_mode] for img in trial_imgs])

        mtom_args = type(
            "Args",
            (),
            {
                "model_name": self.model_name,
                "model": self.model,
                "context_len": self.context_len,
                "tokenizer": self.tokenizer,
                "image_processor": self.image_processor,
                "query": mtom_query,
                "conv_mode": None,
                "sep": ",",
                "images": [merged_context],
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": self.model_args.max_output_tokens,

            },
        )()

        feedback, _ = self.eval_model(mtom_args)
        return feedback.strip()

    
    def query(self, query, trial_imgs, feedback=None):
        query = "".join(query)

        merged_context = merge_images(
            [img[self.model_args.img_mode] for img in trial_imgs]
        )
        args = type(
            "Args",
            (),
            {
                "model_name": self.model_name,
                "model": self.model,
                "context_len": self.context_len,
                "tokenizer": self.tokenizer,
                "image_processor": self.image_processor,
                "query": query,
                "conv_mode": None,
                "sep": ",",
                "images": [merged_context],
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": self.model_args.max_output_tokens,
            },
        )()

        gen_msg, _ = self.eval_model(args)

        if feedback:
            gen_msg = f"{gen_msg} {feedback.strip()}"

        return gen_msg

    def get_spkr_intro(self, context_imgs):
        return [self.model_args.intro_text]

    def _get_spkr_prompt(self, round_name, target_label):
        prompt = [f"\n{round_name}", f"\nTarget: {target_label}. Message:"]
        return prompt

    def get_lsnr_intro(self):
        return [self.model_args.intro_text]

    def _get_lsnr_prompt(self, round_name, trial_imgs, msg, omit_img):
        prompt = [
            f"\n{round_name}",
            f"\nWhich image is this message referring to: {msg}?",
        ]

        return prompt

    def update_with_spkr_pred(self, spkr_trial_prompt, spkr_pred):
        spkr_trial_prompt[-1] = (
            spkr_trial_prompt[-1] + " ASSISTANT: " + spkr_pred + "</s></s>"
        )
        return spkr_trial_prompt

    def update_with_lsnr_pred(self, lsnr_trial_prompt, lsnr_pred):
        lsnr_trial_prompt[-1] = (
            lsnr_trial_prompt[-1] + " ASSISTANT: " + lsnr_pred + "</s></s>"
        )
        return lsnr_trial_prompt

    def _get_spkr_feedback(self, pred_fn, spkr_tgt_img, spkr_trial_imgs):
        if pred_fn == "invalid":
            feedback = "USER: the listener didn't give a valid answer."
        else:
            for i in range(4):
                if spkr_trial_imgs[i]["filename"] == pred_fn:
                    pred_label = self.model_args.label_space[i]
                    break

            if pred_fn == spkr_tgt_img["filename"]:
                feedback = f"USER: the listener correctly answered {pred_label}."
            else:
                feedback = f"USER: the listener mistakenly answered {pred_label}."

        return feedback

    def _get_lsnr_feedback(self, pred, target_label, spkr_msg):
        if pred == target_label:
            feedback = "USER: correct."
        elif pred not in self.model_args.label_space:
            feedback = "USER: invalid answer. Answer must be one of top left, top right, bottom left, bottom right."
        else:
            feedback = f"USER: wrong, {spkr_msg} is referring to {target_label}."

        return feedback


def merge_images(Imgs, padding=2, dim=256):
    collage = Image.new("RGB", (dim * 2 + padding, dim * 2 + padding), color=(0, 0, 0))
    collage.paste(Imgs[0], (0, 0))
    collage.paste(Imgs[1], (dim + 2 * padding, 0))
    collage.paste(Imgs[2], (0, dim + 2 * padding))
    collage.paste(Imgs[3], (dim + 2 * padding, dim + 2 * padding))

    return collage
