from PIL import Image, ImageDraw, ImageFont

def create_combined_image(start_index_png, fastest_index_png, end_index_png, average_response_time_png,
                          velocity_robots_png, start_time, fastest_time, end_time, robot_info_list, output_path):
    # Define the size of each individual PNG image 960 × 720
    image_width = 960
    image_height = 720

    # Load the PNG images
    start_index_image = Image.open(start_index_png)
    fastest_index_image = Image.open(fastest_index_png)
    end_index_image = Image.open(end_index_png)
    average_response_time_image = Image.open(average_response_time_png)
    velocity_robots_image = Image.open(velocity_robots_png)

    # Create a new blank image to hold all the images
    result_width = 3 * image_width  # Three images side by side
    result_height = 2 * image_height  # Two images stacked vertically
    result = Image.new("RGB", (result_width, result_height), "white")

    # Paste the images onto the blank image
    result.paste(start_index_image, (0, 0))
    result.paste(fastest_index_image, (image_width, 0))
    result.paste(end_index_image, (2 * image_width, 0))
    result.paste(average_response_time_image, (0, image_height))
    result.paste(velocity_robots_image, (image_width, image_height))

    # Create a drawing context to add text information
    draw = ImageDraw.Draw(result)
    font = ImageFont.truetype("arial.ttf", 33)  # Specify the font and size

    if fastest_time == end_time:
        same_time = 'True'
    else:
        same_time = 'False'

    # Add text information to the image
    draw.text((10, 10), f"Start Time: {start_time}", fill="black", font=font)
    draw.text((image_width + 10, 10), f"Fastest Time: {fastest_time}", fill="black", font=font)
    draw.text((2 * image_width + 10, 10), f"End Time: {end_time}", fill="black", font=font)
    draw.text((2*image_width + 10, 1.55*image_height + 10), f"End time is the quickest time: {same_time}", fill="black", font=font)

    # Add robot information
    robot_info = "\n\n".join([str(
        "[" + ",    ".join([str(round(element, 3)) for element in info]) + "]"
    ) for info in robot_info_list])
    draw.text((2 * image_width + 10, image_height + 10), "Robot Info [x, y, v ]:\n\n" + robot_info, fill="black", font=font)

    # Save the final image
    result.save(output_path)
