from PIL import Image, ImageDraw, ImageFont

import csv
import json
import imageio
import matplotlib.pyplot as plt
import os


def create_combined_image(start_index_png, end_index_png, average_response_time_png,
                          velocity_robots_png, start_time, end_time, robot_info_list, output_path):
    # Define the size of each individual PNG image 960 × 720
    image_width = 960
    image_height = 720

    # Load the PNG images
    start_index_image = Image.open(start_index_png)
    end_index_image = Image.open(end_index_png)
    average_response_time_image = Image.open(average_response_time_png)
    velocity_robots_image = Image.open(velocity_robots_png)

    # Create a new blank image to hold all the images
    result_width = 3 * image_width  # Three images side by side
    result_height = 2 * image_height  # Two images stacked vertically
    result = Image.new("RGB", (result_width, result_height), "white")

    # Paste the images onto the blank image
    result.paste(start_index_image, (0, 0))
    result.paste(end_index_image, (image_width, 0))
    result.paste(end_index_image, (2 * image_width, 0))
    result.paste(average_response_time_image, (0, image_height))
    result.paste(velocity_robots_image, (image_width, image_height))

    # Create a drawing context to add text information
    draw = ImageDraw.Draw(result)
    font = ImageFont.truetype("arial.ttf", 33)  # Specify the font and size

    # Add text information to the image
    draw.text((10, 10), f"Start Time: {start_time}", fill="black", font=font)
    # draw.text((image_width + 10, 10), f"Fastest Time: {fastest_time}", fill="black", font=font)
    draw.text((2 * image_width + 10, 10), f"End Time: {end_time}", fill="black", font=font)
    # draw.text((2 * image_width + 10, 1.55 * image_height + 10), f"End time is the quickest time: {same_time}",
    #           fill="black", font=font)

    # Add robot information
    robot_info = "\n\n".join([str(
        "[" + ",    ".join([str(round(element, 3)) for element in info]) + "]"
    ) for info in robot_info_list])
    draw.text((2 * image_width + 10, image_height + 10), "Robot Info [x, y, v ]: (at end)\n\n" + robot_info,
              fill="black", font=font)

    # Save the final image
    result.save(output_path)


# def save_data(robots, avg_response_time, p_dot_list, output_path):
#     # save avg_response_time and p_dot_list to csv
#     with open(f'{output_path}/avg_response_time.csv', 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(avg_response_time)
#
#     with open(f'{output_path}/p_dot_list.csv', 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerows(p_dot_list)
#
#     # save start time, end time, quickest time and index of fastest time
#     times_data = {
#         'start_time': avg_response_time[0],
#         'end_time': avg_response_time[-1],
#     }
#
#     with open(f'{output_path}/times.json', 'w') as file:
#         json.dump(times_data, file, indent=4)
#
#     # save the location of the robots with their starting position
#     robots_data = {
#         f"Robot {i + 1}": {
#             "x": robots.return_position(i)[0],
#             "y": robots.return_position(i)[1],
#             "v": robots.return_max_speed(i)
#         }
#         for i in range(robots.number_of_robots())
#     }
#
#     with open(f'{output_path}/robots.json', 'w') as file:
#         json.dump(robots_data, file, indent=4)


def save_gif(images_list, output_path):
    imageio.mimsave(f'{output_path}/gif_speed1.gif', images_list, duration=0.9)
    imageio.mimsave(f'{output_path}/gif_speed2.gif', images_list, duration=0.5)
    imageio.mimsave(f'{output_path}/gif_speed3.gif', images_list, duration=0.3)
    imageio.mimsave(f'{output_path}/gif_speed4.gif', images_list, duration=0.18)


def plot_avg_response_time(avg_response_time_list, output_path=None, log=False, show=True, title='Plot of average response time'):
    indices = list(range(len(avg_response_time_list)))

    plt.figure()
    plt.plot(indices, avg_response_time_list, marker='o')

    plt.title(title)
    plt.xlabel('Iteration')

    plt.grid(True)

    if log:
        plt.yscale('log', base=10)
        plt.ylabel('Cost = Time\u00B2 (seconds\u00B2) log10')  # , labelpad=15
    else:
        plt.ylabel('Cost = Time\u00B2 (seconds\u00B2)')

    # want to change this later, just temporary
    if title == 'Plot of average response time':
        string = "Avg_response_time"
    else:
        string = title.replace(" ", "_")

    if (output_path is not None) and (log==False):
        plt.savefig(f"{output_path}/{string}.png", dpi=150)
    elif (output_path is not None) and (log==True):
        plt.savefig(f"{output_path}/{string}_log.png", dpi=150)

    if show:
        plt.show()


def plot_p_dot_list(p_dot_list, stop_criterion_val, output_path=None, show=True):
    plt.figure()
    p_dot_robots = list(map(list, zip(*p_dot_list)))
    for number, p_dots in enumerate(p_dot_robots):
        plt.plot(p_dots, label=f"Robot {number + 1}")

    plt.axhline(y=stop_criterion_val, color='r', linestyle='-', label='Stop criterion')

    plt.title('Plot of velocity vector of all robots')
    plt.xlabel('Iteration')
    plt.ylabel('Pdot (m/s)')

    plt.legend()
    if output_path is not None:
        plt.savefig(f"{output_path}/Velocity_robots.png", dpi=150)

    if show:
        plt.show()

def compare_loyds_to_mw(response_mw, loyds_response, loyds_response_mw_voro, loyds_response_speed_eq, iteration):
    # Find the longest list
    max_length = max(len(response_mw), len(loyds_response), len(loyds_response_mw_voro), len(loyds_response_speed_eq))

    # Plotting the lists
    plt.figure()
    list_names = ["MW Vor Time\u00B2", "Loyds vor Time\u00B2", "Loyds mw vor Time\u00B2", "Loyds equal speed Time\u00B2"]
    # Iterate over the lists
    for i, lst in enumerate([
        response_mw,
        loyds_response,
        loyds_response_mw_voro, loyds_response_speed_eq
    ]):
        # Plot the values until the length of the current list
        plt.plot(range(len(lst)), lst[:max_length], label=list_names[i])

        # Plot the dotted line for the last value of the list
        plt.plot(len(lst) - 1, lst[-1], 'o', linestyle='dotted', color=f'C{i}')

    # Set the limits of the x-axis and y-axis
    plt.xlim(0, max_length - 1)
    plt.ylim(0, max(max(response_mw), max(loyds_response), max(loyds_response_mw_voro), max(loyds_response_speed_eq)))

    # Add legend and labels
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Value')

    # Show the plot
    plt.savefig(f"compared_all_lists_{iteration}.png", dpi=150)
    plt.show()


def export_data_run(start_positions_and_v, end_positions_and_v, stop_criterion,
                    avg_response_time_mw_vor, p_dot_list, gain, step_dt, run_number,
                    output_path="."):
    data = {
        "start_positions_and_v" : start_positions_and_v,
        "mw_vor_end_positions_and_v" : end_positions_and_v,
        "stop_criterion" : stop_criterion,
        "mw_vor_start_time" : avg_response_time_mw_vor[0],
        "mw_vor_end_time" : avg_response_time_mw_vor[-1],
        "mw_vor_response_time_list" : avg_response_time_mw_vor,
        "p_dot_list" : p_dot_list,
        "gain_for_p_dot" : gain,
        "step_dt" : step_dt,
    }
    json_file_name = f'{output_path}/data_run_{run_number}.json'
    with open(json_file_name, 'w') as file:
        json.dump(data, file, indent=4)

    return json_file_name

def append_lloyds_run_to_data(existing_data_path, end_positions_and_v_lloyds,
                              lloyds_avg_response_time, lloyds_avg_response_time_as_mw,
                              lloyds_avg_response_time_speed_eq, gain_lloyds):
    new_data = {
        "lloyds_end_positions_and_v" : end_positions_and_v_lloyds,
        "lloyds_avg_response_time" : lloyds_avg_response_time,
        "lloyds_avg_response_time_as_mw" : lloyds_avg_response_time_as_mw,
        "lloyds_avg_response_time_speed_eq" : lloyds_avg_response_time_speed_eq,
        "lloyds_start_time_as_mw" : lloyds_avg_response_time_as_mw[0],
        "lloyds_end_time_as_mw" : lloyds_avg_response_time_as_mw[-1],
        "lloyds_start_time_speed_eq" : lloyds_avg_response_time_speed_eq[0],
        "lloyds_end_time_speed_eq" : lloyds_avg_response_time_speed_eq[-1],
        "gain_for_p_dot_lloyds" : gain_lloyds
    }

    with open(existing_data_path, 'r') as file:
        existing_data = json.load(file)

    existing_data.update(new_data)

    with open(existing_data_path, 'w') as file:
        json.dump(existing_data, file, indent=4)

    return None

def export_mesh(x_mesh, y_mesh, z_mesh, output_path):
    data = {
        "size_mesh": [z_mesh.shape[0], z_mesh.shape[1]],
        "x_mesh": x_mesh.tolist(),
        "y_mesh": y_mesh.tolist(),
        "z_mesh": z_mesh.tolist()
    }

    json_file_name = f'{output_path}/global_mesh_data.json'
    with open(json_file_name, 'w') as file:
        json.dump(data, file, indent=4)

    return json_file_name