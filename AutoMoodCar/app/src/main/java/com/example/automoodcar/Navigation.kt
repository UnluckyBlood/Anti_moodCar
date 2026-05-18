package com.example.automoodcar

import androidx.compose.runtime.Composable
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.automoodcar.camera.CameraScreen
import com.example.automoodcar.gallery.GalleryScreen
import com.example.automoodcar.history.HistoryScreen

@Composable
fun Navigation() {
    val navController = rememberNavController()

    NavHost(navController = navController, startDestination = "camera") {
        composable("camera") { CameraScreen(navController) }
        composable("history") { HistoryScreen() }
        composable("gallery") { GalleryScreen() }
    }
}