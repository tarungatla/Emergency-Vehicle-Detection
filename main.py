import time
import os
import threading


class PriorityLaneSystem:
    def __init__(self):
        self.lanes = [
            {"id": 1, "normalTokens": 10, "specialTokens": 5, "cyclesSinceLastUse": 0},
            {"id": 2, "normalTokens": 15, "specialTokens": 10, "cyclesSinceLastUse": 0},
            {"id": 3, "normalTokens": 20, "specialTokens": 8, "cyclesSinceLastUse": 0},
            {"id": 4, "normalTokens": 12, "specialTokens": 12, "cyclesSinceLastUse": 0}
        ]
        self.is_processing = False
        self.current_lane = None
        self.stop_event = threading.Event()

    def has_special_tokens(self):
        """Check if any lane has special tokens."""
        return any(lane["specialTokens"] > 0 for lane in self.lanes)

    def get_priority_lane(self):
        """Get the lane with the highest priority based on token types."""
        if self.has_special_tokens():
            # Sort by special tokens first
            return sorted(self.lanes, key=lambda x: x["specialTokens"], reverse=True)[0]
        else:
            # Sort by normal tokens, prioritizing lanes unused for 2 cycles
            return sorted(self.lanes, key=lambda x: (-1 if x["cyclesSinceLastUse"] >= 2 else 0, x["normalTokens"]),
                          reverse=True)[0]

    def print_lane_status(self):
        """Print the current status of all lanes."""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n🛣️  PRIORITY-BASED LANE SYSTEM 🛣️")
        print("⭐ Special Tokens get priority")
        print("🔵 Normal Tokens processed after special tokens\n")

        for lane in self.lanes:
            lane_id = lane["id"]
            normal_tokens = lane["normalTokens"]
            special_tokens = lane["specialTokens"]

            # Determine if this is the current active lane
            is_active = self.current_lane and self.current_lane["id"] == lane_id

            # Create visual indicator for traffic light
            if is_active:
                traffic_light = "🟢"  # Green for active
            else:
                traffic_light = "🔴"  # Red for inactive

            # Format the line with some basic styling
            line = f"{traffic_light} Lane {lane_id}: "

            # Add highlighting for the active lane
            if is_active:
                line += "→→→ "
            else:
                line += "     "

            line += f"⭐ Special: {special_tokens} | 🔵 Normal: {normal_tokens}"

            # Add additional status for the active lane
            if is_active:
                if special_tokens > 0:
                    line += f" | Processing Special Tokens..."
                else:
                    line += f" | Processing Normal Tokens..."

            print(line)

        print("\nPress Ctrl+C to stop the simulation")

    def reduce_tokens(self, lane):
        """Reduce tokens for a lane during its green light period."""
        reduction_rate_normal = 3  # Normal tokens reduced per second
        reduction_rate_special = 5  # Special tokens reduced per second
        duration = 5  # 5 seconds for green signal
        has_special = lane["specialTokens"] > 0

        # Update cycles for all other lanes
        for other_lane in self.lanes:
            if other_lane["id"] != lane["id"]:
                other_lane["cyclesSinceLastUse"] += 1

        # Reset cycles for current lane
        lane["cyclesSinceLastUse"] = 0

        # Process tokens
        for _ in range(duration):
            if self.stop_event.is_set():
                return

            self.current_lane = lane
            self.print_lane_status()

            if has_special:
                # Reduce special tokens first
                lane["specialTokens"] = max(0, lane["specialTokens"] - reduction_rate_special)
                has_special = lane["specialTokens"] > 0
            else:
                # Reduce normal tokens
                lane["normalTokens"] = max(0, lane["normalTokens"] - reduction_rate_normal)

            time.sleep(1)

        # Yellow light transition (2 seconds)
        self.print_lane_status()
        time.sleep(2)

    def check_next_lane(self):
        """Check and process the next lane based on priority."""
        priority_lane = self.get_priority_lane()
        has_special = priority_lane["specialTokens"] > 0

        print(f"\n🔵 PRIORITY: Lane {priority_lane['id']} ({'Special' if has_special else 'Normal'} Tokens)")

        if priority_lane["specialTokens"] == 0 and priority_lane["normalTokens"] == 0:
            # If the priority lane has no tokens, check if any lane has tokens
            has_any_tokens = any(lane["specialTokens"] > 0 or lane["normalTokens"] > 0 for lane in self.lanes)
            if not has_any_tokens:
                print("\nAll tokens processed!")
                self.is_processing = False
                return False

        self.reduce_tokens(priority_lane)
        return True

    def start_process(self):
        """Start the lane processing."""
        self.is_processing = True

        try:
            while self.is_processing and not self.stop_event.is_set():
                if not self.check_next_lane():
                    break
        except KeyboardInterrupt:
            print("\nSimulation stopped by user.")
            self.stop_event.set()
            self.is_processing = False

    def update_lane_tokens(self, lane_id, normal_tokens, special_tokens):
        """Update token values for a specific lane."""
        for lane in self.lanes:
            if lane["id"] == lane_id:
                lane["normalTokens"] = normal_tokens
                lane["specialTokens"] = special_tokens
                break


def main():
    """Main function to run the program."""
    lane_system = PriorityLaneSystem()

    # Display initial configuration
    lane_system.print_lane_status()
    print("\nDo you want to modify the initial token values? (y/n)")

    if input().lower() == 'y':
        for lane in lane_system.lanes:
            lane_id = lane["id"]
            print(f"\nLane {lane_id}:")
            try:
                normal = int(input(f"  Normal Tokens (current: {lane['normalTokens']}): "))
                special = int(input(f"  Special Tokens (current: {lane['specialTokens']}): "))
                lane_system.update_lane_tokens(lane_id, normal, special)
            except ValueError:
                print("Invalid input, keeping current values.")

    print("\nStarting the process... (Press Ctrl+C to stop)")
    time.sleep(1)

    # Start the processing
    lane_system.start_process()

    print("\nSimulation completed.")


if __name__ == "__main__":
    main()