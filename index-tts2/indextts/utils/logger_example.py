"""
Example usage of the colorful logger system for IndexTTS2
"""

from indextts.utils.logger import get_logger, create_progress

def example_basic_usage():
    """Basic logger usage examples"""
    logger = get_logger("Example")
    
    logger.stage("Basic Logger Examples")
    
    logger.info("This is an info message")
    logger.success("This is a success message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.debug("This is a debug message")
    
    print()
    logger.rule("Separator Example")
    print()

def example_emotion_vector():
    """Example of emotion vector display"""
    logger = get_logger("Example")
    
    logger.stage("Emotion Vector Display")
    
    emotion_dict = {
        "happy": 0.8,
        "angry": 0.1,
        "sad": 0.2,
        "afraid": 0.0,
        "disgusted": 0.0,
        "melancholic": 0.3,
        "surprised": 0.5,
        "calm": 0.2,
    }
    
    logger.print_emotion_vector(emotion_dict)

def example_time_stats():
    """Example of time statistics display"""
    logger = get_logger("Example")
    
    logger.stage("Time Statistics Display")
    
    time_stats = {
        "GPT Generation": 2.5,
        "GPT Forward": 1.2,
        "S2Mel": 3.8,
        "BigVGAN": 1.5,
    }
    
    total_time = sum(time_stats.values())
    audio_length = 5.0
    
    logger.print_time_stats(time_stats, total_time, audio_length)

def example_dict_display():
    """Example of hierarchical dictionary display"""
    logger = get_logger("Example")
    
    logger.stage("Hierarchical Dictionary Display")
    
    config = {
        "model": "IndexTTS2",
        "device": "cuda:0",
        "fp16": True,
        "generation": {
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 30,
        },
        "audio": {
            "sample_rate": 22050,
            "channels": 1,
        }
    }
    
    logger.print_dict("Configuration", config)

def example_progress_bar():
    """Example of progress bar usage"""
    import time
    
    logger = get_logger("Example")
    logger.stage("Progress Bar Example")
    
    # Create progress bar
    with create_progress() as progress:
        task_id = progress.add_task("Processing files", total=100.0)
        
        for i in range(100):
            time.sleep(0.02)  # Simulate work
            progress.update(advance=1, description=f"Processing file {i+1}/100")
    
    logger.success("All files processed!")

def example_panel():
    """Example of panel display"""
    logger = get_logger("Example")
    
    logger.stage("Panel Display Example")
    
    logger.panel(
        "This is an important message displayed in a panel.\n"
        "Panels are great for highlighting key information!",
        title="Important Notice",
        style="yellow"
    )

if __name__ == "__main__":
    example_basic_usage()
    print("\n")
    
    example_emotion_vector()
    print("\n")
    
    example_time_stats()
    print("\n")
    
    example_dict_display()
    print("\n")
    
    example_panel()
    print("\n")
    
    # Progress bar should be last as it's interactive
    example_progress_bar()
