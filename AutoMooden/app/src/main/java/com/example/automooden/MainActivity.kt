package com.example.automooden

import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import com.example.automooden.ui.theme.AutoMoodenTheme

class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "AutoMooden"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        Log.d(TAG, "Приложение запущено")

        setContent {
            AutoMoodenTheme {
                AppNavigator()
            }
        }
    }
}

// -------------------- НАВИГАЦИЯ --------------------

enum class Screen {
    LOGIN,
    MAIN
}

@Composable
fun AppNavigator() {

    var currentScreen by remember {
        mutableStateOf(Screen.LOGIN)
    }

    when (currentScreen) {

        Screen.LOGIN -> {
            LoginScreen(
                onLoginSuccess = {
                    currentScreen = Screen.MAIN
                }
            )
        }

        Screen.MAIN -> {
            MainMenuScreen()
        }
    }
}

// -------------------- AUTH --------------------

object Auth {

    private const val TAG = "AutoMoodenAuth"

    private val users = mapOf(
        "admin" to "admin123",
        "user" to "123"
    )

    fun check(login: String, password: String): Boolean {

        Log.d(TAG, "Попытка входа: $login")

        val success = users[login] == password

        if (success) {
            Log.i(TAG, "Успешный вход")
        } else {
            Log.e(TAG, "Ошибка авторизации")
        }

        return success
    }
}

// -------------------- LOGIN SCREEN --------------------

@Composable
fun LoginScreen(
    onLoginSuccess: () -> Unit
) {

    val context = LocalContext.current

    var login by remember { mutableStateOf("") }
    var password by remember { mutableStateOf("") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),

        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {

        Text(
            text = "AutoMooden",
            style = MaterialTheme.typography.headlineLarge,
            color = MaterialTheme.colorScheme.primary
        )

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "Авторизация",
            style = MaterialTheme.typography.titleLarge
        )

        Spacer(modifier = Modifier.height(32.dp))

        OutlinedTextField(
            value = login,
            onValueChange = { login = it },
            label = { Text("Логин") },
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(8.dp))

        OutlinedTextField(
            value = password,
            onValueChange = { password = it },
            label = { Text("Пароль") },
            visualTransformation = PasswordVisualTransformation(),
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(24.dp))

        Button(
            onClick = {

                if (Auth.check(login.trim(), password.trim())) {

                    Toast.makeText(
                        context,
                        "Успешный вход",
                        Toast.LENGTH_SHORT
                    ).show()

                    Log.i("LoginScreen", "Переход в главное меню")

                    onLoginSuccess()

                } else {

                    Toast.makeText(
                        context,
                        "Неверный логин или пароль",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Войти")
        }
    }
}

// -------------------- ГЛАВНОЕ МЕНЮ --------------------

@Composable
fun MainMenuScreen() {

    val context = LocalContext.current

    var manualNumber by remember {
        mutableStateOf("")
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),

        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {

        Text(
            text = "Главное меню",
            style = MaterialTheme.typography.headlineMedium
        )

        Spacer(modifier = Modifier.height(32.dp))

        // КНОПКА КАМЕРЫ

        Button(
            onClick = {

                Log.d("MainMenu", "Открытие камеры")

                Toast.makeText(
                    context,
                    "Камера будет подключена позже",
                    Toast.LENGTH_SHORT
                ).show()
            },
            modifier = Modifier
                .fillMaxWidth()
                .height(60.dp)
        ) {
            Text("📷 Открыть камеру")
        }

        Spacer(modifier = Modifier.height(24.dp))

        Text(
            text = "Или введите номер вручную"
        )

        Spacer(modifier = Modifier.height(16.dp))

        OutlinedTextField(
            value = manualNumber,
            onValueChange = {
                manualNumber = it.uppercase()
            },
            label = {
                Text("Госномер")
            },
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(20.dp))

        Button(
            onClick = {

                Log.d(
                    "MainMenu",
                    "Введен номер: $manualNumber"
                )

                if (manualNumber.isNotEmpty()) {

                    Toast.makeText(
                        context,
                        "Поиск номера: $manualNumber",
                        Toast.LENGTH_SHORT
                    ).show()

                } else {

                    Toast.makeText(
                        context,
                        "Введите номер",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Открыть профиль")
        }
    }
}