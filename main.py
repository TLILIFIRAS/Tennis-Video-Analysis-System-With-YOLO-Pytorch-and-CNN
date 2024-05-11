from utils.video_utils import (read_video , save_video)
from trackers.player_tracker import PlayerTracker
from trackers.ball_tracker import BallTracker
from court_line_detector import CourtLineDetector
import cv2
from mini_court import MiniCourt
from utils import measure_distance
from copy import deepcopy
import pandas as pd
from utils import draw_player_stats
from utils import convert_meters_to_pixel_distance , convert_pixel_distance_to_meters
import constants
def main():
    input_video_path="input_Videos/input_video.mp4"
    video_frames=read_video(input_video_path)
    #Detecting players and the ball
    player_tracker = PlayerTracker(model_path='./yolov8x.pt')
    ball_tracker = BallTracker(model_path='./models/ball_detector_model.pt')
    
    player_detections = player_tracker.detect_frames(video_frames,read_from_stub=True,stub_path="./tracker_stubs/player_detections.pkl")
    Ball_detections = ball_tracker.detect_frames(video_frames,read_from_stub=True,stub_path="./tracker_stubs/ball_detections.pkl")
    Ball_detections= ball_tracker.interpolate_ball_position(Ball_detections)
    #detect court lines
    court_model_path="./models/tennis_keypoints_detector_model.pth"
    court_line_detector=CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])
    #choose the main two players
    player_detections=player_tracker.choose_and_filter_players(court_keypoints,player_detections)

    #mini court 
    mini_court = MiniCourt(video_frames[0])
    
    #detect ball shots 
    ball_shot_frames=ball_tracker.get_ball_shot_frames(Ball_detections)
    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, Ball_detections ,court_keypoints)
    
    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
    } ]
    
     #Calculating Speed of players , shots 
    for ball_shot_ind in range(len(ball_shot_frames)-1):
        start_frame=ball_shot_frames[ball_shot_ind]
        end_frame=ball_shot_frames[ball_shot_ind+1]
        ball_shot_time_in_seconds=(end_frame - start_frame) /24      
        #get ddistance covered by ball
        distance_covered_by_ball_in_pexels=measure_distance(ball_mini_court_detections[start_frame][1],
                                                            ball_mini_court_detections [end_frame][1])
        distance_covered_by_ball_in_meters=convert_pixel_distance_to_meters( distance_covered_by_ball_in_pexels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           )
        #get the speed of the ball shot in KM/H
        speed_of_ball_shot= distance_covered_by_ball_in_meters/ball_shot_time_in_seconds * 3.6
        
        # player who shot the ball 
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball =min(player_positions.keys() , key=lambda player_id : measure_distance( player_positions[player_id],ball_mini_court_detections[start_frame][1]))
        
        #speed the the opponent player
        opponent_player_id=1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id], 
                                                               player_mini_court_detections[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters=convert_pixel_distance_to_meters( distance_covered_by_opponent_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           ) 
        speed_of_opponent = distance_covered_by_ball_in_meters / ball_shot_time_in_seconds * 3.6
        current_player_stats= deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot
        
        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent
        player_stats_data.append(current_player_stats)
        
    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_1_number_of_shots']


        
        
        
        
    
    #draw bbox
    ##draw player bboxes
    output_video_frames=player_tracker.draw_bboxes(video_frames,player_detections)
    ##draw ball bboxes
    output_video_frames=ball_tracker.draw_bboxes(output_video_frames,Ball_detections)
    ##draw court lines
    output_video_frames=court_line_detector.draw_keypoints_on_video(output_video_frames,court_keypoints)
    ##draw minic court
    output_video_frames=mini_court.draw_mini_court(output_video_frames)
    output_video_frames=mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
    output_video_frames=mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections,color=(23, 123, 255))
    # Draw Player Stats
    output_video_frames = draw_player_stats(output_video_frames,player_stats_data_df)

    
    
    ##draw fps 
    for i , frame in enumerate(output_video_frames):
        cv2.putText(frame,f"FPS:{i}",(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,178),2)
    save_video(output_video_frames,"./Output_Videos/output_video.avi")
    
if __name__ == "__main__":
    main()