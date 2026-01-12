#!/usr/bin/env bash
#
# Focus Music Installer
# https://github.com/cubny/focus
#
# This script installs Focus Music and its dependencies (uv, PortAudio).
# It works on macOS and Linux.
#
# Usage:
#   curl -LsSf https://raw.githubusercontent.com/cubny/focus/main/install.sh | bash
#
# Or download and run:
#   chmod +x install.sh && ./install.sh
#

set -e

# --- Configuration ---
REPO_URL="https://github.com/cubny/focus.git"
INSTALL_DIR="${FOCUS_HOME:-$HOME/.local/share/focus}"
BIN_DIR="${HOME}/.local/bin"
MIN_PYTHON_VERSION="3.12"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# --- Helper Functions ---
info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1" >&2
}

header() {
    echo ""
    echo -e "${BOLD}$1${NC}"
    echo "─────────────────────────────────────"
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# --- OS Detection ---
detect_os() {
    case "$(uname -s)" in
        Darwin*)    OS="macos" ;;
        Linux*)     OS="linux" ;;
        MINGW*|MSYS*|CYGWIN*) 
            error "Windows is not supported by this installer."
            error "Please use WSL (Windows Subsystem for Linux) instead."
            exit 1
            ;;
        *)
            error "Unsupported operating system: $(uname -s)"
            exit 1
            ;;
    esac
}

# --- Dependency Installation ---
install_portaudio() {
    header "Installing PortAudio (audio library)"
    
    if [ "$OS" = "macos" ]; then
        if command_exists brew; then
            info "Installing PortAudio via Homebrew..."
            brew install portaudio || warn "PortAudio may already be installed"
            success "PortAudio installed"
        else
            warn "Homebrew not found. PortAudio wheel includes bundled library."
            info "If you have audio issues, install Homebrew and run: brew install portaudio"
        fi
    elif [ "$OS" = "linux" ]; then
        if command_exists apt-get; then
            info "Installing PortAudio via apt..."
            sudo apt-get update -qq
            sudo apt-get install -y -qq libportaudio2 portaudio19-dev
            success "PortAudio installed"
        elif command_exists dnf; then
            info "Installing PortAudio via dnf..."
            sudo dnf install -y portaudio portaudio-devel
            success "PortAudio installed"
        elif command_exists pacman; then
            info "Installing PortAudio via pacman..."
            sudo pacman -S --noconfirm portaudio
            success "PortAudio installed"
        else
            warn "Could not detect package manager."
            warn "Please install PortAudio manually: https://www.portaudio.com/"
        fi
    fi
}

install_uv() {
    header "Installing uv (Python package manager)"
    
    if command_exists uv; then
        success "uv is already installed: $(uv --version)"
        return 0
    fi
    
    info "Downloading and installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    
    if command_exists uv; then
        success "uv installed: $(uv --version)"
    else
        error "Failed to install uv. Please install manually:"
        error "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
}

install_focus() {
    header "Installing Focus Music"
    
    info "Installing focus-music package..."
    
    # Use uv tool install for global CLI availability
    # --python flag ensures we use Python 3.12+
    if uv tool install --python ">=${MIN_PYTHON_VERSION}" focus-music 2>/dev/null; then
        success "Focus Music installed from PyPI"
    else
        # Fallback: install from git if not on PyPI yet
        info "Installing from GitHub repository..."
        uv tool install --python ">=${MIN_PYTHON_VERSION}" "git+${REPO_URL}"
        success "Focus Music installed from GitHub"
    fi
}

setup_path() {
    header "Setting up PATH"
    
    # Ensure ~/.local/bin is in PATH
    if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
        warn "$BIN_DIR is not in your PATH"
        
        # Detect shell and config file
        SHELL_NAME=$(basename "$SHELL")
        case "$SHELL_NAME" in
            bash)
                SHELL_RC="$HOME/.bashrc"
                ;;
            zsh)
                SHELL_RC="$HOME/.zshrc"
                ;;
            fish)
                SHELL_RC="$HOME/.config/fish/config.fish"
                ;;
            *)
                SHELL_RC="$HOME/.profile"
                ;;
        esac
        
        info "Adding $BIN_DIR to PATH in $SHELL_RC"
        
        if [ "$SHELL_NAME" = "fish" ]; then
            echo "set -gx PATH \"$BIN_DIR\" \$PATH" >> "$SHELL_RC"
        else
            echo "" >> "$SHELL_RC"
            echo "# Added by Focus Music installer" >> "$SHELL_RC"
            echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$SHELL_RC"
        fi
        
        success "PATH updated in $SHELL_RC"
        warn "Run 'source $SHELL_RC' or restart your terminal to use 'focus' command"
    else
        success "PATH is already configured"
    fi
}

print_success() {
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}${BOLD}  🎵 Focus Music installed successfully!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${BOLD}Next steps:${NC}"
    echo ""
    echo "  1. Set your Google API key (get one at https://aistudio.google.com/):"
    echo -e "     ${BLUE}export GOOGLE_API_KEY=\"your-api-key\"${NC}"
    echo ""
    echo "  2. Start a focus session:"
    echo -e "     ${BLUE}focus start --profile deep-work${NC}"
    echo ""
    echo "  3. List available profiles:"
    echo -e "     ${BLUE}focus profiles${NC}"
    echo ""
    echo -e "${BOLD}Documentation:${NC} https://github.com/cubny/focus"
    echo ""
}

# --- Main ---
main() {
    echo ""
    echo -e "${BOLD}🎵 Focus Music Installer${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    detect_os
    info "Detected OS: $OS"
    
    install_portaudio
    install_uv
    install_focus
    setup_path
    print_success
}

main "$@"
