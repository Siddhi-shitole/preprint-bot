from transformers import pipeline

# Load the HuggingFace summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_sections(grobid_output, max_length=130, min_length=30):
    """
    Summarizes the sections of the document extracted by GROBID.

    Args:
        grobid_output (dict): The output dictionary from `extract_grobid_sections`.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.

    Returns:
        dict: A dictionary containing the summarized sections.
    """
    summarized_sections = []

    for section in grobid_output['sections']:
        header = section['header']
        text = section['text']

        # Skip empty sections
        if not text.strip():
            continue

        # Generate a summary for the section
        try:
            summary = summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )[0]['summary_text']
        except Exception as e:
            summary = f"Error summarizing section '{header}': {str(e)}"

        summarized_sections.append({
            'header': header,
            'summary': summary
        })

    return summarized_sections


if __name__ == "__main__":
    # Example usage with the output of `extract_grobid_sections`
    pdf_file = "testGrobid.pdf"
    grobid_output = extract_grobid_sections(pdf_file)

    # Summarize the sections
    summarized_sections = summarize_sections(grobid_output)

    # Write summarized sections to a file
    with open("summarized_output.txt", "w", encoding="utf-8") as f:
        f.write(f"Title: {grobid_output['title']}\n")
        f.write(f"Abstract: {grobid_output['abstract']}\n\n")
        f.write(f"Authors: {', '.join(grobid_output['authors'])}\n")
        f.write(f"Affiliations: {', '.join(grobid_output['affiliations'])}\n")
        f.write(f"Publication Date: {grobid_output['pub_date']}\n\n")

        f.write("Summarized Sections:\n")
        for sec in summarized_sections:
            f.write(f"\n- {sec['header']}:\n{sec['summary']}\n")

        f.write("\nReferences:\n")
        for ref in grobid_output['references']:
            fixed_author_line = ", ".join(ref['authors']).replace(", -", "-")
            f.write(f"- {ref['title']}\n  by {fixed_author_line}\n")