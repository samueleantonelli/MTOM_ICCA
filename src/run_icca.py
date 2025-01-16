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
    args=None,
    
    
):
    random.seed(random_seed)
    random_seeds = random.sample(range(0, 1000), iters)
    trials_Records = trial_entries[:iters]
    condition = args.condition



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
        
        # Load MToM prompts
        mtom_prompts_path = "/mnt/cimec-storage6/users/simone.baratella/GLPCOND/MTOM_ICCA/src/args/mtom_prompts.json"
        with open(mtom_prompts_path, "r") as f:
            mtom_prompts = json.load(f)

        # Get prompts for the selected condition
        condition = args.condition
        if condition not in mtom_prompts:
            raise ValueError(f"Condition '{condition}' not found in mtom_prompts.json")

        selected_prompts = mtom_prompts[condition]

        
        
        #SPEAKER PIPELINE
        # Step 1: Speaker generates the initial message
        if spkr_exp_args.model_type == "llava":
                gen_msg = spkr_model.query(spkr_prompt, spkr_trial_imgs).strip()
        else:
                gen_msg = spkr_model.query(spkr_prompt).strip()

        # Include speaker history in the MToM prompt
        spkr_history = [entry["spkr_trial_record"] for entry in trials_Records[:t]] if t > 0 else []
        spkr_history_flat = list(chain.from_iterable(spkr_history))  # Flatten history

        # Step 2: Speaker MToM model provides feedback on the initial message
        spkr_mtom_prompt = [
            selected_prompts["spkr_mtom_prompt"],
            f"Description: {gen_msg}"
        ] + spkr_history_flat

        speaker_feedback = spkr_mtom_model.query(spkr_mtom_prompt, spkr_trial_imgs).strip()

        # Step 3: Speaker generates a new refined message incorporating MToM feedback
        speaker_MToM_addon_prompt = [
            selected_prompts["spkr_mtom_addon"],
            f"Previous description: {gen_msg}",
            f"Other player Feedback: {speaker_feedback}"
        ] + spkr_history_flat

        refined_msg = spkr_model.query(speaker_MToM_addon_prompt, spkr_trial_imgs).strip()


       
        # Update the speaker trial prompt with the refined message
        spkr_trial_prompt = spkr_model.update_with_spkr_pred(spkr_trial_prompt, refined_msg)

        # Save the refined message
        R_t["spkr_msg"] = refined_msg
        R_t["tgt_label_for_spkr"] = tgt_label_for_spkr
        
        print("\n")
        
        print("---------------------------------------------------------------------- \n")
        
        print(f"Trial {t}")
        print(f"Speaker Initial Message: {gen_msg}")
        print(f"Speaker MToM Feedback: {speaker_feedback}")
        print(f"Speaker Refined Message: {refined_msg}\n")
        




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
            # R_t['lsnr_pred']=lsnr_pred
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
                msg=refined_msg,
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

            
            #LISTENER PIPELINE
            # Step 1: Listener generates an initial prediction
            if lsnr_exp_args.model_type == "llava":
                lsnr_pred = lsnr_model.query(lsnr_prompt, lsnr_trial_imgs_lsnr_view).lower()
            else:
                lsnr_pred = lsnr_model.query(lsnr_prompt).upper()

            # Include listener history in the MToM prompt
            lsnr_history = [entry["lsnr_trial_record"] for entry in trials_Records[:t]] if t > 0 else []
            lsnr_history_flat = list(chain.from_iterable(lsnr_history))  # Flatten history

            # Step 2: Listener MToM model provides feedback on the initial prediction
            lsnr_mtom_prompt = [
            selected_prompts["lsnr_mtom_prompt"],
            f"Speaker description: {refined_msg}",
            f"Listener prediction: {lsnr_pred}"
            ] + lsnr_history_flat
            
            lsnr_feedback = lsnr_mtom_model.query(lsnr_mtom_prompt, lsnr_trial_imgs_lsnr_view).strip()

            # Step 3: Listener generates a new refined prediction incorporating MToM feedback
            
            listener_MToM_addon_prompt = [
            selected_prompts["lsnr_mtom_addon"],
            f"Target image description: {refined_msg}",
            f"Original prediction: {lsnr_pred}",
            f"Feedback: {lsnr_feedback}"
            ] + lsnr_history_flat

            refined_pred = lsnr_model.query(listener_MToM_addon_prompt, lsnr_trial_imgs_lsnr_view).lower()

            # Update the listener trial prompt with the refined prediction
            lsnr_trial_prompt = lsnr_model.update_with_lsnr_pred(lsnr_trial_prompt, refined_pred)

            # Save the refined prediction
            R_t["tgt_label_for_lsnr"] = tgt_label_for_lsnr
            R_t["lsnr_pred"] = refined_pred

            print(f"Listener Initial Prediction: {lsnr_pred}")
            print(f"Listener MToM Feedback: {lsnr_feedback}")
            print(f"Listener Refined Prediction: {refined_pred}")





            try:
                pred_fn = lsnr_trial_imgs[
                    lsnr_model.model_args.label_space.index(lsnr_pred)
                ]["filename"]
            except ValueError:
                pred_fn = "invalid"
            lsnr_feedback_final = lsnr_model.get_lsnr_feedback(
                refined_pred, lsnr_tgt_img, lsnr_trial_imgs, gen_msg
            )
            lsnr_trial_prompt.append(lsnr_feedback_final)
            
            #MODIFICA ----------------------------------------------------------------------------------------------------------------------------

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
                    {"Gen_msg": refined_msg},
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
                    {"Pred_label": refined_pred},
                    {"Tgt_label": tgt_label_for_lsnr},
                ]
            )

        print(" | ".join([f"{k}: {v}" for d in things_to_print for k, v in d.items()]))

        R_t["spkr_trial_record"] = spkr_trial_prompt
        R_t["lsnr_trial_record"] = lsnr_trial_prompt
            
        print("\n")
        print(f"Trial {t} DEBUG \n")

        print("SPEAKER")
        print(f"Speaker Prompt 1: {spkr_prompt}\n")
        print(f"Speaker message 1: {gen_msg}\n\n")
        
        print(f"SpeakerMToM prompt: {spkr_mtom_prompt}\n")
        print(f"SpeakerMToM message: {speaker_feedback}\n\n")
        
        
        print(f"Speaker Prompt 2: {speaker_MToM_addon_prompt}\n")
        print(f"Speaker Refined Message: {refined_msg}\n\n")

        print("LISTENER")
        print(f"Listener Prompt 1: {lsnr_prompt}\n")
        print(f"Listener message 1: {lsnr_pred}\n\n")
        
        print(f"ListenerMToM prompt: {lsnr_mtom_prompt}\n")
        print(f"ListenerMToM message: {lsnr_feedback}\n\n")
        
        
        print(f"Listener Prompt 2: {listener_MToM_addon_prompt}\n")
        print(f"Listener final answer: {refined_pred}\n\n")



        print("---------------------------------------------------------------------- \n")


    return trials_Records, spkr_prompt, lsnr_prompt


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

    #MODIFICA ----------------------------------------------------------------------------------------------------------------------------

    spkr_mtom_model,
    lsnr_mtom_model,

    #MODIFICA ----------------------------------------------------------------------------------------------------------------------------

    img_mask,
    img_mask_url=None,
    img_mask_base64=None,
    sleep_time=0,
    img_hosting_site=None,
    exp_name=None,
    args=None,
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

        records, spkr_hist, lsnr_hist = eval_loop(
            trials_this_context,
            context_imgs,
            iters=num_of_trials,
            random_seed=seeds[i],
            spkr_exp_args=spkr_exp_args,
            lsnr_exp_args=lsnr_exp_args,
            spkr_model=spkr_model,
            lsnr_model=lsnr_model,

            #MODIFICA ----------------------------------------------------------------------------------------------------------------------------

            spkr_mtom_model=spkr_mtom_model,
            lsnr_mtom_model=lsnr_mtom_model,

            #MODIFICA ----------------------------------------------------------------------------------------------------------------------------

            img_mask=img_mask,
            img_mask_url=img_mask_url,
            img_mask_base64=img_mask_base64,
            sleep_time=sleep_time,
            args=args,
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
        full_transcripts = {"spkr_hist": spkr_hist, "lsnr_hist": lsnr_hist}

        spkr_model_args = (
            spkr_model.model_args.__dict__ if spkr_model is not None else None
        )
        lsnr_model_args = (
            lsnr_model.model_args.__dict__ if lsnr_model is not None else None
        )

        report_to_save.update(
            {
                "spkr_model_arg": spkr_model_args,
                "lsnr_model_args": lsnr_model_args,
                "spkr_exp_args": spkr_exp_args.__dict__,
                "lsnr_exp_args": lsnr_exp_args.__dict__,
            }
        )

        os.makedirs(dirname_save, exist_ok=True)
        records_df.to_csv(f"{dirname_save}/records_{dtime}_{save_suffix}_output.csv")
        with open(
            f"{dirname_save}/records_{dtime}_{save_suffix}_report.json", "w"
        ) as f:
            json.dump(report_to_save, f, indent=4)

        with open(
            f"{dirname_save}/records_{dtime}_{save_suffix}_full_transcripts.pickle",
            "wb",
        ) as f:
            pickle.dump(full_transcripts, f)

    return records


def main():
    parser = argparse.ArgumentParser()

    #MODIFICA ----------------------------------------------------------------------------------------------------------------------------

    # Command-line argument, even if we are probably only using one
    parser.add_argument("--spkr_mtom_ckpt", type=str, default="path_to_spkr_mtom_model")
    parser.add_argument("--lsnr_mtom_ckpt", type=str, default="path_to_lsnr_mtom_model")

    #MODIFICA ----------------------------------------------------------------------------------------------------------------------------
    
    parser.add_argument("--condition", type=str, default="base")


    parser.add_argument(
        "--spkr_model_type", type=str, default="llava"
    )  
    parser.add_argument(
        "--lsnr_model_type", type=str, default="llava"
    )
    parser.add_argument(
        "--spkr_intro_version", type=str, default="simple"
    )  
    parser.add_argument("--lsnr_intro_version", type=str, default="standard")
    parser.add_argument(
        "--sleep_time", type=int, default=0
    )  
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="ICCA_data")
    parser.add_argument(
        "--spkr_img_mode", type=str, default="PIL"
    )  # llava uses PIL as the image format
    parser.add_argument("--lsnr_img_mode", type=str, default="PIL")
    parser.add_argument(
        "--spkr_model_ckpt", type=str, default="liuhaotian/llava-v1.6-vicuna-7b"
    )  
    parser.add_argument(
        "--lsnr_model_ckpt", type=str, default="liuhaotian/llava-v1.6-vicuna-7b"
    )
    parser.add_argument(
        "--spkr_intro_texts", type=str, default="args/intro_texts_spkr.json"
    )
    parser.add_argument(
        "--lsnr_intro_texts", type=str, default="args/intro_texts_lsnr.json"
    )
    parser.add_argument(
        "--lsnr_exp_args_fp", type=str, default="args/interaction_args_lsnr.json"
    )
    parser.add_argument(
        "--img_hosting_site", type=str, default=""
    ) 
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument(
        "--num_of_trials", type=int, default=5
    )  # can use smaller numbers for debugging

    args = parser.parse_args()

    with open(args.spkr_intro_texts, "r") as f:
        spkr_intro_texts = json.load(f)

    with open(args.lsnr_intro_texts, "r") as f:
        lsnr_intro_texts = json.load(f)
    

    # Load MToM-specific prompts
    with open(args.spkr_intro_texts, "r") as f:
        spkr_mtom_intro_texts = json.load(f)

    with open(args.lsnr_intro_texts, "r") as f:
        lsnr_mtom_intro_texts = json.load(f)




    # Extract the relevant prompts for the scenario
    spkr_intro_text = spkr_intro_texts["llava"][args.spkr_intro_version]
    lsnr_intro_text = lsnr_intro_texts["llava"][args.lsnr_intro_version]
    spkr_mtom_intro_text = spkr_mtom_intro_texts["llava"][args.spkr_intro_version]
    lsnr_mtom_intro_text = lsnr_mtom_intro_texts["llava"][args.lsnr_intro_version]

    with open(args.lsnr_exp_args_fp, "r") as f:
        lsnr_exp_args = json.load(f)

    spkr_exp_args = InteractionArgs(model_type=args.spkr_model_type)
    lsnr_exp_args = InteractionArgs(model_type=args.lsnr_model_type, **lsnr_exp_args)


    print("initializing spkr model..")
    match spkr_exp_args.model_type:
        case "llava":
            spkr_model_args = ModelArgs(
                role="spkr",
                model_ckpt=args.spkr_model_ckpt,
                img_mode=args.spkr_img_mode,
                max_output_tokens=30,
                label_space=["top left", "top right", "bottom left", "bottom right"],
                intro_text=spkr_intro_text,
            )
            spkr_model = LlavaModel(spkr_model_args)

        case "Human":
            spkr_model = None

    print("initializing lsnr model..")
    match lsnr_exp_args.model_type:

        case "llava":
            lsnr_model_args = ModelArgs(
                role="lsnr",
                model_ckpt=args.lsnr_model_ckpt,
                max_output_tokens=5,
                img_mode=args.lsnr_img_mode,
                label_space=["top left", "top right", "bottom left", "bottom right"],
                intro_text=lsnr_intro_text,
            )
            lsnr_model = LlavaModel(lsnr_model_args)
            
        case "oracle":
            lsnr_model = None

    #MODIFICA ----------------------------------------------------------------------------------------------------------------------------

    # Initialize MToM Models, like the others
    print("initializing MToM models...")
    spkr_mtom_model = LlavaModel(
        ModelArgs(
            role="spkr_mtom",
            model_ckpt="liuhaotian/llava-v1.6-vicuna-7b",
            max_output_tokens=30,
            img_mode="PIL",
            label_space=["top left", "top right", "bottom left", "bottom right"],
            intro_text=spkr_mtom_intro_text
        )
    )

    lsnr_mtom_model = LlavaModel(
        ModelArgs(
            role="lsnr_mtom",
            model_ckpt="liuhaotian/llava-v1.6-vicuna-7b",
            max_output_tokens=30,
            img_mode="PIL",
            label_space=["top left", "top right", "bottom left", "bottom right"],
            intro_text=lsnr_mtom_intro_text
    )
    )

    #MODIFICA ----------------------------------------------------------------------------------------------------------------------------

    if lsnr_exp_args.img_mask:
        img_mask = Image.new("RGB", (256, 256), color=(0, 0, 0))
        buffer = io.BytesIO()
        img_mask.save(buffer, format="JPEG")
        buffer.seek(0)
        img_mask = Image.open(buffer)
        img_mask_url = f"{args.img_hosting_site}/black_mask.jpg"
        img_mask_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    else:
        img_mask = None
        img_mask_url = None
        img_mask_base64 = None

    print(
        f"spkr: {spkr_exp_args.model_type} | lsnr: {lsnr_exp_args.model_type} | spkr_intro: {args.spkr_intro_version} | lsnr_intro: {args.lsnr_intro_version}"
    )
    save_suffix = f"{spkr_exp_args.model_type}_{args.spkr_intro_version}_{lsnr_exp_args.model_type}_{args.lsnr_intro_version}"

    test_dir = args.data_dir
    context_records_fps = glob.glob(f"{test_dir}/*/trials_for*.csv")
    context_records_fps.sort()

    dtime = datetime.now().strftime("%Y%m%d-%H%M%S")

    run_test(
        context_records_fps,
        num_of_trials=args.num_of_trials,
        random_seed=args.seed,
        save_suffix=save_suffix,
        spkr_exp_args=spkr_exp_args,
        dtime=dtime,
        lsnr_exp_args=lsnr_exp_args,
        spkr_model=spkr_model,
        lsnr_model=lsnr_model,
        spkr_mtom_model=spkr_mtom_model,
        lsnr_mtom_model=lsnr_mtom_model,
        img_mask=img_mask,
        img_mask_url=img_mask_url,
        img_mask_base64=img_mask_base64,
        sleep_time=args.sleep_time,
        img_hosting_site=args.img_hosting_site,
        exp_name=args.exp_name,
        args=args,
    )


if __name__ == "__main__":
    main()
