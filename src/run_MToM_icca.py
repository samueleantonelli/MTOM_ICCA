# %%
import pandas as pd
import glob, os, io, random, json, time, argparse, copy, base64, pickle
from datetime import datetime
from PIL import Image
from utils import *
from tqdm import tqdm
from sklearn.metrics import classification_report
from MLLMs import *

def eval_loop(
    trial_entries,
    context_imgs,
    iters,
    random_seed,
    spkr_exp_args,
    lsnr_exp_args,
    spkr_model,
    lsnr_model,
    spkr_mtom_model,
    lsnr_mtom_model,
    img_mask,
    img_mask_url=None,
    img_mask_base64=None,
    sleep_time=0,
):
    random.seed(random_seed)
    random_seeds = random.sample(range(0, 1000), iters)
    trials_Records = trial_entries[:iters]

    # prepare the instruction that appears in the beginning of the interaction
    spkr_intro = (
        spkr_model.get_spkr_intro(context_imgs)
        if spkr_exp_args.model_type != "Human"
        else []
    )
    lsnr_intro = (
        lsnr_model.get_lsnr_intro() if lsnr_exp_args.model_type != "oracle" else []
    )

    spkr_context_imgs = context_imgs
    if lsnr_exp_args.img_mask:
        lsnr_context_imgs = copy.deepcopy(context_imgs)
        lsnr_context_imgs = masking_images(
            lsnr_context_imgs, img_mask, img_mask_url, img_mask_base64
        )
    else:
        lsnr_context_imgs = context_imgs.copy()
        random.shuffle(
            lsnr_context_imgs
        )  # speaker and listener see the images in potentially different orders

    for t, R_t in enumerate(trials_Records):
        if t != 0:
            time.sleep(sleep_time)
        tgt_fn = R_t["targetImg"]
        human_msg = R_t["msg"]

        if spkr_exp_args.model_type == "Human":
            gen_msg = human_msg
            spkr_trial_prompt = []
            spkr_prompt = []
        else:
            (
                spkr_prompt,
                spkr_trial_prompt,
                spkr_tgt_img,
                tgt_label_for_spkr,
                spkr_trial_imgs,
            ) = spkr_model.get_spkr_prompt(
                spkr_intro,
                t,
                spkr_context_imgs,
                tgt_fn,
                spkr_exp_args,
                records=trials_Records,
            )

            R_t["spkr_trial_fns"] = [img["filename"] for img in spkr_trial_imgs]

            ##############
            # query the speaker
            if spkr_exp_args.model_type == "llava":
                gen_msg = spkr_model.query(spkr_prompt, spkr_trial_imgs).strip()
            else:
                gen_msg = spkr_model.query(spkr_prompt).strip()

            # Integrate feedback from speaker MToM
            spkr_feedback = spkr_mtom_model.query(
                ["Feedback prompt for MToM"], spkr_trial_imgs
            )
            spkr_trial_prompt = spkr_model.update_with_spkr_pred(
                spkr_trial_prompt, gen_msg + spkr_feedback
            )

            R_t["spkr_msg"] = gen_msg
            R_t["tgt_label_for_spkr"] = tgt_label_for_spkr

        #############
        # listener prompt prep
        if lsnr_exp_args.model_type == "oracle":
            pred_fn, lsnr_trial_prompt, R_t["tgt_label_for_lsnr"], R_t["lsnr_pred"] = (
                spkr_tgt_img["filename"],
                [],
                tgt_label_for_spkr,
                tgt_label_for_spkr,
            )
            lsnr_prompt = []
        else:
            omit_img = True if (lsnr_exp_args.img_once and t > 0) else False

            if t in lsnr_exp_args.misleading_trials:
                misleading = True
                omit_img = False  # override the omit_img
            else:
                misleading = False

            (
                lsnr_prompt,
                lsnr_trial_prompt,
                lsnr_tgt_img,
                tgt_label_for_lsnr,
                lsnr_trial_imgs,
                lsnr_trial_imgs_lsnr_view,
            ) = lsnr_model.get_lsnr_prompt(
                lsnr_intro,
                t,
                lsnr_context_imgs,
                tgt_fn,
                msg=gen_msg,
                records=trials_Records,
                random_seed=random_seeds[t],
                no_history=lsnr_exp_args.no_history,
                do_shuffle=lsnr_exp_args.do_shuffle,
                omit_img=omit_img,
                misleading=misleading,
                has_intro=lsnr_exp_args.has_intro,
            )
            # lsnr_trial_imgs_lsnr_view is different from lsnr_trial_imgs for experiments that have misleading manipulations

            R_t["lsnr_trial_fns"] = [img["filename"] for img in lsnr_trial_imgs]

            ##############
            # query the listener
            if lsnr_exp_args.model_type == "llava":
                lsnr_pred = lsnr_model.query(
                    lsnr_prompt, lsnr_trial_imgs_lsnr_view
                ).lower()
            else:
                lsnr_pred = lsnr_model.query(lsnr_prompt).upper()

            lsnr_pred = lsnr_pred.strip()

            # Integrate feedback from listener MToM
            lsnr_feedback = lsnr_mtom_model.query(
                ["Feedback prompt for MToM"], lsnr_trial_imgs_lsnr_view
            )
            lsnr_trial_prompt = lsnr_model.update_with_lsnr_pred(
                lsnr_trial_prompt, lsnr_pred + lsnr_feedback
            )

            R_t["tgt_label_for_lsnr"] = tgt_label_for_lsnr
            R_t["lsnr_pred"] = lsnr_pred

            try:
                pred_fn = lsnr_trial_imgs[
                    lsnr_model.model_args.label_space.index(lsnr_pred)
                ]["filename"]
            except ValueError:
                pred_fn = "invalid"
            listener_feedback = lsnr_model.get_lsnr_feedback(
                lsnr_pred, lsnr_tgt_img, lsnr_trial_imgs, gen_msg
            )
            lsnr_trial_prompt.append(listener_feedback)

        # get speaker feedback
        if spkr_exp_args.model_type != "Human":
            spkr_feedback = spkr_model.get_spkr_feedback(
                pred_fn, spkr_tgt_img, spkr_trial_imgs
            )
            spkr_trial_prompt.append(spkr_feedback)

        things_to_print = []

        if spkr_exp_args.model_type != "Human":
            things_to_print.extend(
                [
                    {"Gen_msg": gen_msg},
                    {"Human_msg": human_msg},
                    {"Tgt_fn": spkr_tgt_img["filename"]},
                ]
            )
        else:
            things_to_print.extend([{"Human_msg": human_msg}, {"Tgt_fn": tgt_fn}])

        if lsnr_exp_args.model_type != "oracle":
            things_to_print.extend(
                [
                    {"Pred_fn": pred_fn},
                    {"Pred_label": lsnr_pred},
                    {"Tgt_label": tgt_label_for_lsnr},
                ]
            )

        print(" | ".join([f"{k}: {v}" for d in things_to_print for k, v in d.items()]))

        R_t["spkr_trial_record"] = spkr_trial_prompt
        R_t["lsnr_trial_record"] = lsnr_trial_prompt

    return trials_Records

def run_test(
    context_records_fps,
    num_of_trials,
    random_seed,
    save_suffix,
    dtime,
    spkr_exp_args,
    lsnr_exp_args,
    spkr_model,
    lsnr_model,
    spkr_mtom_model,
    lsnr_mtom_model,
    img_mask,
    img_mask_url=None,
    img_mask_base64=None,
    sleep_time=0,
    img_hosting_site=None,
    exp_name=None,
):
    random.seed(random_seed)
    seeds = random.sample(range(0, 1000), len(context_records_fps))

    for i, context_record_fp in tqdm(enumerate(context_records_fps)):
        datafile_name = os.path.basename(context_record_fp)

        dirname = os.path.dirname(context_record_fp)
        dirname_save = os.path.join("outputs_icca", dirname)

        print(f"Working on context {datafile_name}")
        trials_this_context = pd.read_csv(context_record_fp, index_col=0)

        img_fps = glob.glob(f"{dirname}/COCO*.jpg")
        if len(img_fps) == 0:
            img_fps = glob.glob(f"{dirname}/*.png")
        context_imgs = []

        for fp in img_fps:
            img = Image.open(fp)
            with open(fp, "rb") as image_file:
                binary_data = image_file.read()
                base_64_encoded_data = base64.b64encode(binary_data)
                base64_string = base_64_encoded_data.decode("utf-8")

            fn = os.path.basename(fp)

            if (
                spkr_exp_args.model_type != "Human"
                and spkr_model.model_args.img_mode == "URL"
            ) or (
                lsnr_exp_args.model_type != "oracle"
                and lsnr_model.model_args.img_mode == "URL"
            ):
                url = f"{img_hosting_site}/{fn}"
            else:
                url = None

            entry = {
                "filename": fn,
                "PIL": img,
                "URL": url,
                "base64_string": base64_string,
            }
            context_imgs.append(entry)

        trials_this_context = trials_this_context.to_json(orient="records")
        trials_this_context = json.loads(trials_this_context)

        random.seed(i)
        random.shuffle(context_imgs)

        records = eval_loop(
            trials_this_context,
            context_imgs,
            iters=num_of_trials,
            random_seed=seeds[i],
            spkr_exp_args=spkr_exp_args,
            lsnr_exp_args=lsnr_exp_args,
            spkr_model=spkr_model,
            lsnr_model=lsnr_model,
            spkr_mtom_model=spkr_mtom_model,
            lsnr_mtom_model=lsnr_mtom_model,
            img_mask=img_mask,
            img_mask_url=img_mask_url,
            img_mask_base64=img_mask_base64,
            sleep_time=sleep_time,
        )

        records_df = pd.DataFrame(records)
        print(
            classification_report(
                records_df["tgt_label_for_lsnr"],
                records_df["lsnr_pred"],
                zero_division=0,
            )
        )
        report_to_save = classification_report(
            records_df["tgt_label_for_lsnr"],
            records_df["lsnr_pred"],
            zero_division=0,
            output_dict=True,
        )

        os.makedirs(dirname_save, exist_ok=True)
        records_df.to_csv(f"{dirname_save}/records_{dtime}_{save_suffix}_output.csv")
        with open(
            f"{dirname_save}/records_{dtime}_{save_suffix}_report.json", "w"
        ) as f:
            json.dump(report_to_save, f, indent=4)

    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spkr_model_type", type=str, default="llava")
    parser.add_argument("--lsnr_model_type", type=str, default="llava")
    parser.add_argument("--spkr_intro_version", type=str, default="simple")
    parser.add_argument("--lsnr_intro_version", type=str, default="standard")
    parser.add_argument("--sleep_time", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="ICCA_data")
    parser.add_argument("--spkr_img_mode", type=str, default="PIL")
    parser.add_argument("--lsnr_img_mode", type=str, default="PIL")
    parser.add_argument("--spkr_model_ckpt", type=str, default="liuhaotian/llava-v1.6-vicuna-7b")
    parser.add_argument("--lsnr_model_ckpt", type=str, default="liuhaotian/llava-v1.6-vicuna-7b")
    parser.add_argument("--num_of_trials", type=int, default=24)

    args = parser.parse_args()

    spkr_model_args = ModelArgs(
        role="spkr",
        model_ckpt=args.spkr_model_ckpt,
        max_output_tokens=30,
        label_space=["top left", "top right", "bottom left", "bottom right"],
    )
    spkr_model = LlavaModel(spkr_model_args)

    lsnr_model_args = ModelArgs(
        role="lsnr",
        model_ckpt=args.lsnr_model_ckpt,
        max_output_tokens=5,
        label_space=["top left", "top right", "bottom left", "bottom right"],
    )
    lsnr_model = LlavaModel(lsnr_model_args)

    spkr_mtom_model = LlavaModel(spkr_model_args)  # Placeholder for MToM speaker
    lsnr_mtom_model = LlavaModel(lsnr_model_args)  # Placeholder for MToM listener

    context_records_fps = glob.glob(f"{args.data_dir}/*/trials_for*.csv")
    context_records_fps.sort()

    dtime = datetime.now().strftime("%Y%m%d-%H%M%S")

    run_test(
        context_records_fps=context_records_fps,
        num_of_trials=args.num_of_trials,
        random_seed=args.seed,
        save_suffix="mtom",
        dtime=dtime,
        spkr_exp_args=InteractionArgs(model_type="llava"),
        lsnr_exp_args=InteractionArgs(model_type="llava"),
        spkr_model=spkr_model,
        lsnr_model=lsnr_model,
        spkr_mtom_model=spkr_mtom_model,
        lsnr_mtom_model=lsnr_mtom_model,
        img_mask=None,
    )

if __name__ == "__main__":
    main()
