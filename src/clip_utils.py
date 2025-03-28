from transformers import CLIPProcessor, CLIPModel

def clip_classifier(im, candidate_captions, decimals=4):
    # Load the model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Move model to the specified device
    device = 0
    model = model.to(device)

    # Prepare the image for the model
    inputs = preprocess(images=im, return_tensors="pt").to(device)

    # Generate image features
    image_features = model.get_image_features(**inputs)

    # Prepare text inputs
    text_inputs = preprocess(text=candidate_captions, padding=True, return_tensors="pt").to(device)

    # Get text features
    text_features = model.get_text_features(**text_inputs)

    # Calculate similarity scores
    similarity = (image_features @ text_features.T).squeeze()

    # Get probabilities and sort them from highest to lowest
    probs = similarity.softmax(dim=0)
    sorted_indices = probs.argsort(descending=True)
    sorted_labels = {
        candidate_captions[i]: round(probs[i].item(), decimals)
        for i in sorted_indices
    }
    return sorted_labels
